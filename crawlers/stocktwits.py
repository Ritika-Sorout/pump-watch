"""
StockTwits crawler.

Fetches the public symbol stream for a ticker via the StockTwits REST API.
No authentication required; rate limit is ~200 requests / hour.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import config
from crawlers.base import BaseCrawler
from models import Post

logger = logging.getLogger(__name__)

_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


class StockTwitsCrawler(BaseCrawler):
    PLATFORM = "stocktwits"

    # ── Core ──────────────────────────────────────────────────────

    def crawl(self, ticker: str) -> list[Post]:
        """Paginate through the public stream for *ticker*."""
        posts: list[Post] = []
        url = _STREAM_URL.format(symbol=ticker)
        cursor: int | None = None

        while len(posts) < self.max_posts:
            params: dict = {}
            if cursor is not None:
                params["max"] = cursor

            try:
                resp = self._request("GET", url, params=params)
                data = resp.json()
            except Exception:
                logger.exception("StockTwits request failed for %s", ticker)
                break

            messages = data.get("messages", [])
            if not messages:
                logger.debug("No more messages for %s", ticker)
                break

            for msg in messages:
                post = self._normalise(msg, ticker)
                if post is not None:
                    posts.append(post)

            # Cursor for next page: use the *lowest* message ID
            cursor = messages[-1].get("id")
            if cursor is None:
                break

            # StockTwits returns pages of 30; stop if we got a short page
            if len(messages) < 20:
                break

        return posts[: self.max_posts]

    # ── Normalisation ─────────────────────────────────────────────

    def _normalise(self, msg: dict, ticker: str) -> Post | None:
        """Convert a StockTwits message dict to a Post."""
        try:
            text = msg.get("body", "")
            user = msg.get("user", {})

            # Account age
            account_age = None
            join_date_str = user.get("join_date")
            if join_date_str:
                try:
                    join_dt = datetime.strptime(join_date_str, "%Y-%m-%dT%H:%M:%SZ")
                    join_dt = join_dt.replace(tzinfo=timezone.utc)
                    account_age = Post.compute_account_age_days(join_dt)
                except (ValueError, TypeError):
                    pass

            # Timestamp
            created_at = msg.get("created_at", "")
            try:
                ts = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                ts = ts.replace(tzinfo=timezone.utc)
                timestamp = ts.isoformat()
            except (ValueError, TypeError):
                timestamp = created_at

            # Likes — nested under msg.likes.total or just 0
            likes_data = msg.get("likes", {})
            if isinstance(likes_data, dict):
                likes = likes_data.get("total", 0)
            else:
                likes = 0

            return Post(
                post_id=f"stocktwits_{msg['id']}",
                author_id=str(user.get("id", user.get("username", "unknown"))),
                account_age_days=account_age,
                text=text,
                timestamp_utc=timestamp,
                platform=self.PLATFORM,
                ticker=ticker,
                upvotes=likes,
                cashtags=self.extract_cashtags(text),
            )
        except Exception:
            logger.exception("Failed to normalise StockTwits message")
            return None
