"""
Coordination graph builder.

Loads crawled JSONL posts for a ticker, embeds texts with
sentence-transformers, and creates a directed graph where an edge
author_A → author_B exists when:

  1. cosine_similarity(text_A, text_B) >= threshold  (default 0.85)
  2. abs(timestamp_A - timestamp_B)   <= delta_t     (default 15 min)

Direction: the author who posted *first* points to the author who
posted later (potential coordination / copying signal).

Outputs:
  - ``data/graphs/{ticker}_{date}.gpickle``   NetworkX DiGraph (pickle)
  - ``data/graphs/summary.csv``               Aggregated stats
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger(__name__)


# ── Data loading helpers ──────────────────────────────────────────────


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a tz-aware datetime."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_posts_for_ticker(
    ticker: str, data_dir: str = config.DATA_DIR
) -> list[dict[str, Any]]:
    """Load every JSONL record whose filename matches *ticker*.

    Returns a list of dicts (the raw schema from the crawler).
    """
    raw_dir = Path(data_dir)
    if not raw_dir.exists():
        logger.warning("Data directory %s does not exist", raw_dir)
        return []

    posts: list[dict[str, Any]] = []
    pattern = f"{ticker}_*"
    for path in sorted(raw_dir.glob(pattern)):
        if not path.suffix == ".jsonl":
            continue
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    posts.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Bad JSON at %s:%d", path.name, line_no)

    logger.info("Loaded %d posts for ticker=%s from %s", len(posts), ticker, raw_dir)
    return posts


# ── Graph builder ─────────────────────────────────────────────────────


class GraphBuilder:
    """Build a coordination DiGraph from crawled JSONL posts."""

    def __init__(
        self,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
        time_window_minutes: int = config.TIME_WINDOW_MINUTES,
        embedding_model: str = config.EMBEDDING_MODEL,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        self.batch_size = batch_size

        # Lazy-load model so import is fast
        self._model_name = embedding_model
        self._model = None

    @property
    def model(self):
        """Lazily initialise the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ── Public API ────────────────────────────────────────────────

    def build(self, ticker: str, posts: list[dict] | None = None) -> nx.DiGraph:
        """Build and return the coordination graph for *ticker*.

        If *posts* is ``None`` they are loaded from ``data/raw/``.
        """
        if posts is None:
            posts = load_posts_for_ticker(ticker)

        if len(posts) < 2:
            logger.info("Fewer than 2 posts for %s — returning empty graph", ticker)
            return nx.DiGraph()

        # Deduplicate by post_id (safety net)
        seen_ids: set[str] = set()
        unique_posts: list[dict] = []
        for p in posts:
            pid = p.get("post_id", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique_posts.append(p)
        posts = unique_posts

        # Parse timestamps
        timestamps = [_parse_timestamp(p["timestamp_utc"]) for p in posts]

        # Embed texts
        texts = [p.get("text", "") for p in posts]
        embeddings = self._embed(texts)

        # Build similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        # Build graph
        G = self._build_graph(posts, timestamps, sim_matrix, ticker)

        logger.info(
            "Graph for %s: %d nodes, %d edges, %d weakly-connected components",
            ticker,
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.number_weakly_connected_components(G),
        )
        return G

    def build_and_save(self, ticker: str, posts: list[dict] | None = None) -> Path:
        """Build, save to gpickle, update summary CSV, return path."""
        G = self.build(ticker, posts)
        return self._save(G, ticker)

    # ── Internals ─────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into dense vectors."""
        logger.info("Encoding %d texts (batch_size=%d) …", len(texts), self.batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-norm → dot == cosine
        )
        return embeddings

    def _build_graph(
        self,
        posts: list[dict],
        timestamps: list[datetime],
        sim_matrix: np.ndarray,
        ticker: str,
    ) -> nx.DiGraph:
        """Create the DiGraph from the similarity matrix + time constraint."""
        G = nx.DiGraph(ticker=ticker)
        n = len(posts)

        # ── Aggregate node-level stats ────────────────────────────
        author_posts: dict[str, list[dict]] = defaultdict(list)
        for p in posts:
            author_posts[p["author_id"]].append(p)

        # Add nodes (one per unique author)
        for author_id, aposts in author_posts.items():
            # Most common platform for this author
            platform_counts: dict[str, int] = defaultdict(int)
            ages: list[int] = []
            for ap in aposts:
                platform_counts[ap.get("platform", "unknown")] += 1
                age = ap.get("account_age_days")
                if age is not None:
                    ages.append(age)

            primary_platform = max(platform_counts, key=platform_counts.get)
            avg_age = int(sum(ages) / len(ages)) if ages else None

            G.add_node(
                author_id,
                author_id=author_id,
                account_age_days=avg_age,
                post_count=len(aposts),
                platform=primary_platform,
            )

        # ── Add edges ─────────────────────────────────────────────
        edge_count = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # Time check first (cheap)
                dt = abs((timestamps[i] - timestamps[j]).total_seconds())
                if dt > self.time_window.total_seconds():
                    continue

                # Similarity check
                sim = float(sim_matrix[i, j])
                if sim < self.similarity_threshold:
                    continue

                # Direction: earlier author → later author
                if timestamps[i] <= timestamps[j]:
                    src = posts[i]["author_id"]
                    dst = posts[j]["author_id"]
                else:
                    src = posts[j]["author_id"]
                    dst = posts[i]["author_id"]

                if src == dst:
                    continue  # no self-loops

                # If edge already exists, keep the higher weight
                if G.has_edge(src, dst):
                    existing = G[src][dst]["weight"]
                    if sim > existing:
                        G[src][dst]["weight"] = sim
                else:
                    G.add_edge(src, dst, weight=sim)
                    edge_count += 1

        logger.debug("Created %d directed edges", edge_count)
        return G

    # ── Persistence ───────────────────────────────────────────────

    def _save(self, G: nx.DiGraph, ticker: str) -> Path:
        """Save graph as .gpickle and append to summary.csv."""
        out_dir = Path(config.GRAPH_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # ── gpickle (Python pickle) ───────────────────────────────
        gpickle_path = out_dir / f"{ticker}_{today}.gpickle"
        with open(gpickle_path, "wb") as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved graph → %s", gpickle_path)

        # ── Summary CSV ───────────────────────────────────────────
        csv_path = out_dir / "summary.csv"
        num_clusters = nx.number_weakly_connected_components(G) if G else 0
        row = {
            "ticker": ticker,
            "date": today,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_clusters": num_clusters,
        }

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Update existing row for same ticker+date, or append
            mask = (df["ticker"] == ticker) & (df["date"] == today)
            if mask.any():
                for col, val in row.items():
                    df.loc[mask, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_csv(csv_path, index=False)
        logger.info("Updated summary → %s", csv_path)

        return gpickle_path
