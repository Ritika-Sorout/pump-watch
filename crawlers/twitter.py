"""
Twitter / X crawler using API v2.

Uses the ``/2/tweets/search/recent`` endpoint (available on Basic+
tier) with Bearer-token auth.  Filters by ``$TICKER`` cashtags.

Note: The Academic Research tier (``/search/all``) was retired in 2023.
This implementation targets the **recent search** endpoint which returns
tweets from the last 7 days.  Swap the URL constant to ``/search/all``
if you have elevated access.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import config
from crawlers.base import BaseCrawler
from models import Post

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


class TwitterCrawler(BaseCrawler):
    PLATFORM = "twitter"

    def __init__(self, max_posts: int = config.MAX_POSTS_PER_SOURCE) -> None:
        super().__init__(max_posts=max_posts)
        self._bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        if not self._bearer_token:
            logger.warning(
                "TWITTER_BEARER_TOKEN not set — Twitter crawler will be skipped"
            )

    @property
    def is_configured(self) -> bool:
        return bool(self._bearer_token)

    # ── Core ──────────────────────────────────────────────────────

    def crawl(self, ticker: str) -> list[Post]:
        """Search recent tweets for *$ticker* cashtag."""
        if not self.is_configured:
            logger.info("Twitter crawler not configured — skipping %s", ticker)
            return []

        posts: list[Post] = []
        next_token: str | None = None
        headers = {"Authorization": f"Bearer {self._bearer_token}"}

        while len(posts) < self.max_posts:
            params: dict = {
                "query": f"${ticker} lang:en -is:retweet",
                "max_results": min(100, self.max_posts - len(posts)),
                "tweet.fields": "id,author_id,text,created_at,public_metrics",
                "expansions": "author_id",
                "user.fields": "id,username,created_at",
            }
            if next_token:
                params["next_token"] = next_token

            try:
                resp = self._request("GET", _SEARCH_URL, params=params, headers=headers)
                data = resp.json()
            except Exception:
                logger.exception("Twitter request failed for %s", ticker)
                break

            # Build author lookup from includes
            author_map: dict[str, dict] = {}
            for user in data.get("includes", {}).get("users", []):
                author_map[user["id"]] = user

            tweets = data.get("data", [])
            if not tweets:
                break

            for tweet in tweets:
                post = self._normalise(tweet, author_map, ticker)
                if post is not None:
                    posts.append(post)

            # Pagination
            meta = data.get("meta", {})
            next_token = meta.get("next_token")
            if not next_token:
                break

        return posts[: self.max_posts]

    # ── Normalisation ─────────────────────────────────────────────

    def _normalise(
        self, tweet: dict, author_map: dict[str, dict], ticker: str
    ) -> Post | None:
        """Convert a v2 tweet object to a Post."""
        try:
            text = tweet.get("text", "")
            author_id = tweet.get("author_id", "unknown")
            author_info = author_map.get(author_id, {})

            # Account age
            account_age = None
            user_created = author_info.get("created_at")
            if user_created:
                try:
                    dt = datetime.fromisoformat(user_created.replace("Z", "+00:00"))
                    account_age = Post.compute_account_age_days(dt)
                except (ValueError, TypeError):
                    pass

            # Tweet timestamp
            created_at = tweet.get("created_at", "")
            try:
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                timestamp = ts.isoformat()
            except (ValueError, TypeError):
                timestamp = created_at

            metrics = tweet.get("public_metrics", {})
            likes = metrics.get("like_count", 0)

            return Post(
                post_id=f"twitter_{tweet['id']}",
                author_id=author_info.get("username", author_id),
                account_age_days=account_age,
                text=text,
                timestamp_utc=timestamp,
                platform=self.PLATFORM,
                ticker=ticker,
                upvotes=likes,
                cashtags=self.extract_cashtags(text),
            )
        except Exception:
            logger.exception("Failed to normalise tweet")
            return None
