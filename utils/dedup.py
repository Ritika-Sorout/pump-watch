"""
Persistent deduplication store.

Tracks post IDs that have already been collected so that repeated
crawler runs never write the same post twice.  One store per platform.

Storage: ``data/.dedup_{platform}.json`` — a JSON array of strings.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DeduplicationStore:
    """Set-backed dedup store with JSON persistence."""

    def __init__(self, platform: str, dedup_dir: str = "data") -> None:
        self._platform = platform
        self._path = Path(dedup_dir) / f".dedup_{platform}.json"
        self._seen: set[str] = set()
        self._dirty = False
        self._load()

    # ── Public API ────────────────────────────────────────────────

    def is_seen(self, post_id: str) -> bool:
        """Return ``True`` if *post_id* was already collected."""
        return post_id in self._seen

    def mark_seen(self, post_id: str) -> None:
        """Record *post_id* so future runs skip it."""
        if post_id not in self._seen:
            self._seen.add(post_id)
            self._dirty = True

    def save(self) -> None:
        """Flush to disk (only if changed)."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(sorted(self._seen), fh)
        logger.debug("Saved %d IDs to %s", len(self._seen), self._path)
        self._dirty = False

    @property
    def count(self) -> int:
        return len(self._seen)

    # ── Context manager ───────────────────────────────────────────

    def __enter__(self) -> "DeduplicationStore":
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.save()

    # ── Internals ─────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            logger.debug("No dedup file at %s — starting fresh", self._path)
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._seen = set(data)
            logger.info(
                "Loaded %d known IDs for platform=%s", len(self._seen), self._platform
            )
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt dedup file %s — resetting", self._path)
            self._seen = set()
