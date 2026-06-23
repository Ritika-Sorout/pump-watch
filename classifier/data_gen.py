"""
Training data generator for the linguistic fingerprinter.

Produces a balanced binary dataset:
  label 0 — genuine human-written financial posts (Reddit, pre-GPT era)
  label 1 — synthetic AI-generated due-diligence posts (OpenAI API)

The final dataset is saved as a HuggingFace ``DatasetDict`` (train/test)
under ``data/training/fingerprinter/``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ── Prompt templates for AI-generated posts ──────────────────────────

_AI_PROMPTS: list[str] = [
    (
        "Write a convincing due diligence post about ${TICKER} as if it will 10x. "
        "Include technical analysis, fundamental catalysts, and a bullish thesis. "
        "Write in the casual style of a retail investor on Reddit."
    ),
    (
        "Write a short Reddit-style post hyping ${TICKER} stock. "
        "Mention short squeeze potential, institutional buying, and upcoming catalysts. "
        "Use informal language with some emojis and internet slang."
    ),
    (
        "Create a financial analysis post recommending ${TICKER}. "
        "Discuss revenue growth, market cap, and upcoming earnings. "
        "Write as if you are posting on r/wallstreetbets."
    ),
    (
        "Write a bearish analysis of ${TICKER} warning investors to sell. "
        "Mention overvaluation, insider selling, and debt concerns. "
        "Write in a casual Reddit discussion style."
    ),
    (
        "Compose a short post about ${TICKER} for a financial forum. "
        "Include your position, price target, and reasoning. "
        "Write naturally as a retail trader sharing their thesis."
    ),
]

TRAINING_DIR = "data/training/fingerprinter"


class DatasetGenerator:
    """Collect human posts and generate AI posts for classifier training."""

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        ai_samples_per_ticker: int = config.DATAGEN_AI_SAMPLES_PER_TICKER,
        human_posts_per_sub: int = config.DATAGEN_HUMAN_POSTS_PER_SUBREDDIT,
    ) -> None:
        self.tickers = tickers or config.DATAGEN_TICKERS
        self.ai_samples_per_ticker = ai_samples_per_ticker
        self.human_posts_per_sub = human_posts_per_sub

    # ── Public API ────────────────────────────────────────────────

    def generate(self, output_dir: str = TRAINING_DIR) -> Path:
        """Create the full training dataset and return its path.

        Returns the path to a JSON file containing the train/test data.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        human_texts = self._collect_human_posts()
        ai_texts = self._generate_ai_posts()

        logger.info(
            "Dataset: %d human posts, %d AI posts", len(human_texts), len(ai_texts)
        )

        # Balance dataset
        min_size = min(len(human_texts), len(ai_texts))
        if min_size == 0:
            raise RuntimeError(
                "No training data collected. Check API keys / connectivity."
            )

        random.shuffle(human_texts)
        random.shuffle(ai_texts)
        human_texts = human_texts[:min_size]
        ai_texts = ai_texts[:min_size]

        # Combine and shuffle
        data = [{"text": t, "label": 0} for t in human_texts] + [
            {"text": t, "label": 1} for t in ai_texts
        ]
        random.shuffle(data)

        # Train/test split
        split_idx = int(len(data) * (1 - config.TRAIN_EVAL_SPLIT))
        dataset = {"train": data[:split_idx], "test": data[split_idx:]}

        path = out / "dataset.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Saved dataset → %s  (train=%d, test=%d)",
            path,
            len(dataset["train"]),
            len(dataset["test"]),
        )
        return path

    # ── Human data: Reddit scrape ─────────────────────────────────

    def _collect_human_posts(self) -> list[str]:
        """Scrape real posts from r/wallstreetbets (and related subs).

        Falls back to loading from a local JSONL file at
        ``data/training/human_posts.jsonl`` if PRAW is unavailable
        or credentials are missing.
        """
        # Try local file first
        local_path = Path(TRAINING_DIR) / "human_posts.jsonl"
        if local_path.exists():
            return self._load_local_human(local_path)

        # Try PRAW
        try:
            return self._scrape_reddit_human()
        except Exception:
            logger.exception(
                "Failed to scrape Reddit for human posts. "
                "Provide data/training/fingerprinter/human_posts.jsonl instead."
            )
            return []

    def _scrape_reddit_human(self) -> list[str]:
        """Use PRAW to scrape top posts from financial subreddits."""
        import praw

        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "fin-social-crawler/1.0"),
        )

        texts: list[str] = []
        subs = ["wallstreetbets", "stocks", "investing"]

        for sub_name in subs:
            logger.info("Scraping r/%s for human training data …", sub_name)
            subreddit = reddit.subreddit(sub_name)
            try:
                for submission in subreddit.top(
                    time_filter="all", limit=self.human_posts_per_sub
                ):
                    text = (submission.selftext or "").strip()
                    if len(text) >= 100:  # meaningful posts only
                        texts.append(text[:2000])  # truncate very long posts
            except Exception:
                logger.exception("Error scraping r/%s", sub_name)

        logger.info("Collected %d human posts from Reddit", len(texts))

        # Cache for future runs
        cache_path = Path(TRAINING_DIR) / "human_posts.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            for t in texts:
                fh.write(json.dumps({"text": t}) + "\n")

        return texts

    def _load_local_human(self, path: Path) -> list[str]:
        """Load cached human posts from a JSONL file."""
        texts = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if len(text) >= 50:
                    texts.append(text)
        logger.info("Loaded %d human posts from %s", len(texts), path)
        return texts

    # ── AI data: OpenAI generation ────────────────────────────────

    def _generate_ai_posts(self) -> list[str]:
        """Generate synthetic DD posts using the OpenAI API.

        Falls back to a local cache at
        ``data/training/fingerprinter/ai_posts.jsonl``.
        """
        # Try local cache first
        local_path = Path(TRAINING_DIR) / "ai_posts.jsonl"
        if local_path.exists():
            return self._load_local_ai(local_path)

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set — cannot generate AI training data. "
                "Provide data/training/fingerprinter/ai_posts.jsonl instead."
            )
            return []

        try:
            return self._call_openai(api_key)
        except Exception:
            logger.exception("Failed to generate AI posts via OpenAI")
            return []

    def _call_openai(self, api_key: str) -> list[str]:
        """Call OpenAI API to generate synthetic financial posts."""
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        texts: list[str] = []

        for ticker in self.tickers:
            for i in range(self.ai_samples_per_ticker):
                prompt_template = _AI_PROMPTS[i % len(_AI_PROMPTS)]
                prompt = prompt_template.replace("${TICKER}", ticker)

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a retail investor posting on a "
                                    "financial forum. Write naturally and "
                                    "convincingly."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=500,
                        temperature=0.9,
                    )
                    text = response.choices[0].message.content.strip()
                    if text:
                        texts.append(text)
                        logger.debug(
                            "Generated AI post %d/%d for %s",
                            i + 1,
                            self.ai_samples_per_ticker,
                            ticker,
                        )
                except Exception:
                    logger.exception(
                        "OpenAI generation failed for %s (sample %d)", ticker, i
                    )

                # Basic rate limiting
                time.sleep(0.5)

        logger.info("Generated %d AI posts via OpenAI", len(texts))

        # Cache for future runs
        cache_path = Path(TRAINING_DIR) / "ai_posts.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            for t in texts:
                fh.write(json.dumps({"text": t}) + "\n")

        return texts

    def _load_local_ai(self, path: Path) -> list[str]:
        """Load cached AI-generated posts from a JSONL file."""
        texts = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    texts.append(text)
        logger.info("Loaded %d AI posts from %s", len(texts), path)
        return texts
