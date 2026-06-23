"""
Canonical data model for a social-media post.

Every platform crawler normalises its raw data into this schema
before it is written to JSONL.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True, slots=True)
class Post:
    """A single social-media post normalised to a common schema."""

    post_id: str
    author_id: str
    account_age_days: Optional[int]
    text: str
    timestamp_utc: str          # ISO 8601
    platform: str               # "reddit" | "stocktwits" | "twitter"
    ticker: str                 # The ticker this was collected for
    upvotes: int
    cashtags: list[str] = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict suitable for ``json.dumps``."""
        return asdict(self)

    def to_json(self) -> str:
        """Return a single-line JSON string (one JSONL record)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # ── Factory helpers ───────────────────────────────────────────

    @staticmethod
    def compute_account_age_days(
        created_utc: Optional[float | datetime],
    ) -> Optional[int]:
        """Days between *created_utc* and now.  Returns ``None`` if unknown."""
        if created_utc is None:
            return None
        if isinstance(created_utc, (int, float)):
            created = datetime.fromtimestamp(created_utc, tz=timezone.utc)
        else:
            created = created_utc
        delta = datetime.now(timezone.utc) - created
        return max(delta.days, 0)
