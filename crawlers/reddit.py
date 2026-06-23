"""
Reddit crawler using PRAW.

Searches r/wallstreetbets, r/stocks, r/investing for posts mentioning a
given ticker symbol and normalises them into the common Post schema.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import praw
from praw.models import Submission

import config
from crawlers.base import BaseCrawler
from models import Post

logger = logging.getLogger(__name__)


class RedditCrawler(BaseCrawler):
    PLATFORM = "reddit"

    def __init__(self, max_posts: int = config.MAX_POSTS_PER_SOURCE) -> None:
        super().__init__(max_posts=max_posts)
        self._reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "fin-social-crawler/1.0"),
        )
        logger.info("PRAW initialised (read-only=%s)", self._reddit.read_only)

    # ── Core ──────────────────────────────────────────────────────

    def crawl(self, ticker: str) -> list[Post]:
        """Search target subreddits for *ticker* mentions."""
        posts: list[Post] = []
        per_sub_limit = max(self.max_posts // len(config.REDDIT_SUBREDDITS), 10)

        for sub_name in config.REDDIT_SUBREDDITS:
            subreddit = self._reddit.subreddit(sub_name)
            query = f"${ticker} OR {ticker}"

            logger.debug(
                "Searching r/%s for '%s' (limit=%d)", sub_name, query, per_sub_limit
            )

            try:
                results = subreddit.search(
                    query, sort="new", time_filter="week", limit=per_sub_limit
                )
                for submission in results:
                    post = self._normalise(submission, ticker)
                    if post is not None:
                        posts.append(post)
            except Exception:
                logger.exception("Error searching r/%s for %s", sub_name, ticker)

            if len(posts) >= self.max_posts:
                break

        return posts[: self.max_posts]

    # ── Normalisation ─────────────────────────────────────────────

    def _normalise(self, sub: Submission, ticker: str) -> Post | None:
        """Convert a PRAW Submission to a Post, or None on error."""
        try:
            text = sub.selftext or sub.title
            author_name = str(sub.author) if sub.author else "[deleted]"

            # Account age — author may be suspended / deleted
            account_age = None
            try:
                if sub.author and hasattr(sub.author, "created_utc"):
                    # Force-load the Redditor to get created_utc
                    account_age = Post.compute_account_age_days(sub.author.created_utc)
            except Exception:
                logger.debug("Could not fetch account age for %s", author_name)

            return Post(
                post_id=f"reddit_{sub.id}",
                author_id=author_name,
                account_age_days=account_age,
                text=text,
                timestamp_utc=datetime.fromtimestamp(
                    sub.created_utc, tz=timezone.utc
                ).isoformat(),
                platform=self.PLATFORM,
                ticker=ticker,
                upvotes=sub.score,
                cashtags=self.extract_cashtags(text),
            )
        except Exception:
            logger.exception("Failed to normalise submission %s", getattr(sub, "id", "?"))
            return None
