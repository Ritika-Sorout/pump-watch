#!/usr/bin/env python3
"""
fin-social-crawler — Multi-source financial social-media data collector.

Usage
-----
    # Crawl all configured tickers on all platforms
    python main.py

    # Override tickers
    python main.py --tickers GME TSLA NVDA

    # Specific platforms only
    python main.py --platforms reddit stocktwits

    # Limit posts per source
    python main.py --max-posts 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import config
from crawlers.base import BaseCrawler
from crawlers.reddit import RedditCrawler
from crawlers.stocktwits import StockTwitsCrawler
from crawlers.twitter import TwitterCrawler

# ── Logging ───────────────────────────────────────────────────────────

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FMT, level=level, stream=sys.stderr)
    # Quiet noisy libraries
    logging.getLogger("prawcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl financial social media for ticker mentions."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=config.TICKERS,
        help="Ticker symbols to crawl (default: %(default)s)",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["reddit", "stocktwits", "twitter"],
        choices=["reddit", "stocktwits", "twitter"],
        help="Platforms to crawl (default: all)",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=config.MAX_POSTS_PER_SOURCE,
        help="Max posts per ticker per platform (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


# ── Crawler factory ───────────────────────────────────────────────────


def build_crawlers(
    platforms: list[str], max_posts: int
) -> dict[str, BaseCrawler]:
    """Instantiate only the requested platform crawlers."""
    registry: dict[str, type[BaseCrawler]] = {
        "reddit": RedditCrawler,
        "stocktwits": StockTwitsCrawler,
        "twitter": TwitterCrawler,
    }
    crawlers: dict[str, BaseCrawler] = {}
    for name in platforms:
        cls = registry.get(name)
        if cls is None:
            logging.warning("Unknown platform: %s", name)
            continue
        try:
            crawler = cls(max_posts=max_posts)
            # Skip Twitter if not configured
            if hasattr(crawler, "is_configured") and not crawler.is_configured:
                logging.info("Skipping %s (not configured)", name)
                continue
            crawlers[name] = crawler
        except Exception:
            logging.exception("Failed to initialise %s crawler", name)
    return crawlers


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("main")
    tickers: list[str] = [t.upper() for t in args.tickers]
    logger.info("Tickers: %s", tickers)
    logger.info("Platforms: %s", args.platforms)

    crawlers = build_crawlers(args.platforms, args.max_posts)
    if not crawlers:
        logger.error("No crawlers available — check your .env credentials.")
        sys.exit(1)

    # Ensure output directory exists
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)

    # ── Crawl loop ────────────────────────────────────────────────
    summary: dict[str, dict[str, int]] = {}
    t0 = time.monotonic()

    for ticker in tickers:
        summary[ticker] = {}
        for platform_name, crawler in crawlers.items():
            logger.info("━━━ Crawling %s / %s ━━━", platform_name, ticker)
            try:
                new_posts = crawler.run(ticker)
                summary[ticker][platform_name] = len(new_posts)
            except Exception:
                logger.exception("Crawler failed: %s / %s", platform_name, ticker)
                summary[ticker][platform_name] = -1  # signals error

    elapsed = time.monotonic() - t0

    # ── Summary ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("  CRAWL COMPLETE  (%.1fs)", elapsed)
    logger.info("═" * 60)
    for ticker, platforms in summary.items():
        parts = []
        for pname, count in platforms.items():
            label = f"{count} new" if count >= 0 else "ERROR"
            parts.append(f"{pname}={label}")
        logger.info("  %-6s  %s", ticker, "  ".join(parts))
    logger.info("═" * 60)
    logger.info("Output: %s/", config.DATA_DIR)


if __name__ == "__main__":
    main()
