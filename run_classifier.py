#!/usr/bin/env python3
"""
run_classifier — Run the linguistic fingerprinter on crawled data.

Reads JSONL files from data/raw/, classifies each post as AI or human,
and writes enriched output to data/analyzed/{ticker}_{date}_bert.jsonl.

Usage
-----
    # Process all configured tickers
    python run_classifier.py

    # Specific tickers
    python run_classifier.py --tickers GME TSLA

    # Custom threshold
    python run_classifier.py --threshold 0.6

    # Custom model directory
    python run_classifier.py --model-dir models/fingerprinter/final
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import config

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FMT, level=level, stream=sys.stderr)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AI-generated post detection on crawled JSONL data."
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=config.TICKERS,
        help="Ticker symbols to process (default: %(default)s)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=config.AI_PROB_THRESHOLD,
        help="AI probability threshold (default: %(default)s)",
    )
    p.add_argument(
        "--model-dir",
        default=str(__import__("pathlib").Path(config.CLASSIFIER_DIR) / "final"),
        help="Path to fine-tuned model (default: %(default)s)",
    )
    p.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help="Input JSONL directory (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default=config.ANALYZED_DIR,
        help="Output directory (default: %(default)s)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("run_classifier")

    tickers = [t.upper() for t in args.tickers]
    logger.info("Tickers: %s", tickers)
    logger.info("Threshold: %.2f", args.threshold)
    logger.info("Model: %s", args.model_dir)

    from classifier.predictor import LinguisticFingerprinter

    fingerprinter = LinguisticFingerprinter(
        model_dir=args.model_dir,
        threshold=args.threshold,
    )

    t0 = time.monotonic()
    results = fingerprinter.process_all_tickers(
        tickers=tickers,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    elapsed = time.monotonic() - t0

    # ── Summary ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("  CLASSIFICATION COMPLETE  (%.1fs)", elapsed)
    logger.info("═" * 60)
    for ticker, path in results.items():
        if path:
            logger.info("  %-6s → %s", ticker, path)
        else:
            logger.info("  %-6s   (no posts found)", ticker)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
