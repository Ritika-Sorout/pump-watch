#!/usr/bin/env python3
"""
build_graph — Coordination graph builder for crawled social-media data.

Reads JSONL files from ``data/raw/``, computes pairwise text similarity
with sentence-transformers, and creates directed graphs connecting
authors whose posts are suspiciously similar within a short time window.

Usage
-----
    # Build graphs for all configured tickers
    python build_graph.py

    # Specific tickers
    python build_graph.py --tickers GME TSLA

    # Custom thresholds
    python build_graph.py --threshold 0.80 --window 30

    # Verbose output
    python build_graph.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import config
from graph.builder import GraphBuilder

# ── Logging ───────────────────────────────────────────────────────────

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FMT, level=level, stream=sys.stderr)
    # Quiet noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build coordination graphs from crawled financial social-media data."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=config.TICKERS,
        help="Ticker symbols to process (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for edges (default: %(default)s)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=config.TIME_WINDOW_MINUTES,
        help="Max time window in minutes between posts (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=config.EMBEDDING_MODEL,
        help="Sentence-transformer model name (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help="Directory containing raw JSONL files (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("build_graph")

    tickers = [t.upper() for t in args.tickers]
    logger.info("Tickers: %s", tickers)
    logger.info(
        "Params: threshold=%.2f  window=%d min  model=%s",
        args.threshold,
        args.window,
        args.model,
    )

    # ── Verify data exists ────────────────────────────────────────
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(
            "Data directory %s does not exist.  Run the crawler first (python main.py).",
            data_path,
        )
        sys.exit(1)

    builder = GraphBuilder(
        similarity_threshold=args.threshold,
        time_window_minutes=args.window,
        embedding_model=args.model,
    )

    # ── Build loop ────────────────────────────────────────────────
    t0 = time.monotonic()
    results: dict[str, dict] = {}

    for ticker in tickers:
        logger.info("━━━ Building graph: %s ━━━", ticker)
        try:
            gpickle_path = builder.build_and_save(ticker)
            # Re-load to get stats (we just built it)
            import pickle

            with open(gpickle_path, "rb") as fh:
                G = pickle.load(fh)
            import networkx as nx

            results[ticker] = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "clusters": nx.number_weakly_connected_components(G),
                "path": str(gpickle_path),
            }
        except Exception:
            logger.exception("Failed to build graph for %s", ticker)
            results[ticker] = {"error": True}

    elapsed = time.monotonic() - t0

    # ── Summary ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("  GRAPH BUILD COMPLETE  (%.1fs)", elapsed)
    logger.info("═" * 60)
    for ticker, info in results.items():
        if "error" in info:
            logger.info("  %-6s  ERROR", ticker)
        else:
            logger.info(
                "  %-6s  nodes=%-4d  edges=%-4d  clusters=%-3d  → %s",
                ticker,
                info["nodes"],
                info["edges"],
                info["clusters"],
                info["path"],
            )
    logger.info("═" * 60)
    logger.info("Summary: %s/summary.csv", config.GRAPH_DIR)


if __name__ == "__main__":
    main()
