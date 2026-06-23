"""
Abstract base crawler.

Provides:
- Rate-limit-aware HTTP requests with exponential backoff on 429
- JSONL file writer (appends to ``data/raw/{ticker}_{platform}_{date}.jsonl``)
- Cashtag extraction from text via regex
- Deduplication integration
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

import config
from models import Post
from utils import DeduplicationStore

logger = logging.getLogger(__name__)

_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b", re.IGNORECASE)


class BaseCrawler(ABC):
    """Abstract crawler that all platform implementations extend."""

    PLATFORM: str = ""  # Override in subclass

    def __init__(self, max_posts: int = config.MAX_POSTS_PER_SOURCE) -> None:
        self.max_posts = max_posts
        self._session = requests.Session()
        self._dedup: Optional[DeduplicationStore] = None

    # ── Public entry point ────────────────────────────────────────

    def run(self, ticker: str) -> list[Post]:
        """Crawl *ticker* with dedup, write JSONL, return new posts."""
        with DeduplicationStore(self.PLATFORM, config.DEDUP_DIR) as dedup:
            self._dedup = dedup
            posts = self.crawl(ticker)
            new_posts = [p for p in posts if not dedup.is_seen(p.post_id)]

            if new_posts:
                self._write_jsonl(ticker, new_posts)
                for p in new_posts:
                    dedup.mark_seen(p.post_id)

            skipped = len(posts) - len(new_posts)
            logger.info(
                "[%s/%s] collected=%d  new=%d  dupes_skipped=%d",
                self.PLATFORM,
                ticker,
                len(posts),
                len(new_posts),
                skipped,
            )
            self._dedup = None
            return new_posts

    @abstractmethod
    def crawl(self, ticker: str) -> list[Post]:
        """Fetch posts for *ticker*.  Must be implemented by subclasses."""
        ...

    # ── Rate-limited HTTP ─────────────────────────────────────────

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        max_retries: int = 5,
    ) -> requests.Response:
        """Issue an HTTP request with automatic retry on 429 / 5xx."""
        backoff = config.BACKOFF_BASE_SECONDS

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.request(
                    method, url, params=params, headers=headers, timeout=30
                )
            except requests.RequestException as exc:
                logger.warning(
                    "[%s] Request error (attempt %d/%d): %s",
                    self.PLATFORM,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt == max_retries:
                    raise
                self._sleep_backoff(backoff)
                backoff = min(backoff * 2, config.BACKOFF_MAX_SECONDS)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                # Honour Retry-After header if present
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff
                else:
                    wait = backoff

                logger.warning(
                    "[%s] HTTP %d — backing off %.1fs (attempt %d/%d)",
                    self.PLATFORM,
                    resp.status_code,
                    wait,
                    attempt,
                    max_retries,
                )
                if attempt == max_retries:
                    resp.raise_for_status()
                self._sleep_backoff(wait)
                backoff = min(backoff * 2, config.BACKOFF_MAX_SECONDS)
                continue

            resp.raise_for_status()
            return resp

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Exhausted retries")  # pragma: no cover

    # ── JSONL writer ──────────────────────────────────────────────

    def _write_jsonl(self, ticker: str, posts: list[Post]) -> Path:
        """Append *posts* to the daily JSONL file and return its path."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = Path(config.DATA_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{ticker}_{self.PLATFORM}_{today}.jsonl"

        with open(path, "a", encoding="utf-8") as fh:
            for post in posts:
                fh.write(post.to_json() + "\n")

        logger.debug("Wrote %d records to %s", len(posts), path)
        return path

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def extract_cashtags(text: str) -> list[str]:
        """Return unique uppercase cashtags found in *text*."""
        return sorted(set(m.upper() for m in _CASHTAG_RE.findall(text)))

    @staticmethod
    def _sleep_backoff(base: float) -> None:
        """Sleep *base* seconds ± jitter."""
        jitter = base * config.BACKOFF_JITTER_FACTOR
        duration = base + random.uniform(-jitter, jitter)
        time.sleep(max(duration, 0.1))
