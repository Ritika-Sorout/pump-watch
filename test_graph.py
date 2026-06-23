#!/usr/bin/env python3
"""
Integration test for the coordination graph builder.

Creates synthetic JSONL data with known coordination patterns,
runs the graph builder, and validates the output.
"""

import json
import os
import pickle
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))


def write_synthetic_data(data_dir: str) -> None:
    """Write JSONL with 3 coordinated posts + 1 unrelated post."""
    os.makedirs(data_dir, exist_ok=True)
    posts = [
        # Cluster 1: Two near-identical posts 5 minutes apart
        {
            "post_id": "reddit_aaa",
            "author_id": "user_alpha",
            "account_age_days": 30,
            "text": "GME is going to the moon, buy now before the short squeeze!",
            "timestamp_utc": "2026-06-23T12:00:00+00:00",
            "platform": "reddit",
            "ticker": "GME",
            "upvotes": 100,
            "cashtags": ["GME"],
        },
        {
            "post_id": "stocktwits_bbb",
            "author_id": "user_beta",
            "account_age_days": 5,
            "text": "GME going to the moon! Buy before the short squeeze happens!",
            "timestamp_utc": "2026-06-23T12:05:00+00:00",
            "platform": "stocktwits",
            "ticker": "GME",
            "upvotes": 50,
            "cashtags": ["GME"],
        },
        # Same cluster: another copy 10 min after alpha
        {
            "post_id": "twitter_ccc",
            "author_id": "user_gamma",
            "account_age_days": 2,
            "text": "GME is heading to the moon, buy now before the squeeze!",
            "timestamp_utc": "2026-06-23T12:10:00+00:00",
            "platform": "twitter",
            "ticker": "GME",
            "upvotes": 20,
            "cashtags": ["GME"],
        },
        # Unrelated post — different topic, 1 hour later
        {
            "post_id": "reddit_ddd",
            "author_id": "user_delta",
            "account_age_days": 1000,
            "text": "Just finished my taxes. Anyone know a good accountant in NYC?",
            "timestamp_utc": "2026-06-23T13:00:00+00:00",
            "platform": "reddit",
            "ticker": "GME",
            "upvotes": 3,
            "cashtags": [],
        },
        # Semantically similar but 2 hours later (outside time window)
        {
            "post_id": "reddit_eee",
            "author_id": "user_epsilon",
            "account_age_days": 15,
            "text": "GME going to the moon, buy before the short squeeze!",
            "timestamp_utc": "2026-06-23T14:05:00+00:00",
            "platform": "reddit",
            "ticker": "GME",
            "upvotes": 10,
            "cashtags": ["GME"],
        },
    ]
    path = os.path.join(data_dir, "GME_test_2026-06-23.jsonl")
    with open(path, "w") as fh:
        for p in posts:
            fh.write(json.dumps(p) + "\n")
    print(f"✓ Wrote {len(posts)} synthetic posts to {path}")


def main() -> None:
    import config

    # Use a temp directory for isolation
    tmp = tempfile.mkdtemp(prefix="graph_test_",
                           dir=os.path.dirname(__file__))
    raw_dir = os.path.join(tmp, "raw")
    graph_dir = os.path.join(tmp, "graphs")

    # Temporarily override config
    orig_data_dir = config.DATA_DIR
    orig_graph_dir = config.GRAPH_DIR
    config.DATA_DIR = raw_dir
    config.GRAPH_DIR = graph_dir

    try:
        write_synthetic_data(raw_dir)

        from graph.builder import GraphBuilder

        builder = GraphBuilder(
            similarity_threshold=0.85,
            time_window_minutes=15,
        )

        # Build
        gpickle_path = builder.build_and_save("GME")
        print(f"✓ Graph saved to {gpickle_path}")

        # Load and validate
        with open(gpickle_path, "rb") as fh:
            G = pickle.load(fh)

        import networkx as nx

        print(f"\n  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Clusters (WCC): {nx.number_weakly_connected_components(G)}")

        # ── Assertions ────────────────────────────────────────────
        print()

        # user_delta should have no edges (unrelated topic)
        if "user_delta" in G:
            assert G.degree("user_delta") == 0, "user_delta should have no edges"
            print("✓ user_delta (unrelated) has no edges")
        else:
            print("✓ user_delta (unrelated) not even in graph — correct")

        # user_epsilon: similar text but 2h later → outside 15min window
        if "user_epsilon" in G:
            assert G.degree("user_epsilon") == 0, (
                "user_epsilon should have no edges (outside time window)"
            )
            print("✓ user_epsilon (time-gapped) has no edges")
        else:
            print("✓ user_epsilon (time-gapped) not in graph — correct")

        # The coordinated cluster should have edges among alpha, beta, gamma
        coord_authors = {"user_alpha", "user_beta", "user_gamma"}
        edges_among_coord = [
            (u, v) for u, v in G.edges() if u in coord_authors and v in coord_authors
        ]
        assert len(edges_among_coord) > 0, "Expected coordination edges"
        print(f"✓ Found {len(edges_among_coord)} coordination edges among α/β/γ")

        # Edges should be directed from earlier → later
        for u, v in edges_among_coord:
            w = G[u][v]["weight"]
            assert w >= 0.85, f"Edge weight {w} below threshold"
            print(f"    {u} → {v}  weight={w:.3f}")

        # Node attributes
        for node in G.nodes():
            data = G.nodes[node]
            assert "author_id" in data, f"Missing author_id on {node}"
            assert "post_count" in data, f"Missing post_count on {node}"
            assert "platform" in data, f"Missing platform on {node}"

        print("✓ Node attributes present")

        # Summary CSV
        import pandas as pd

        csv_path = os.path.join(graph_dir, "summary.csv")
        assert os.path.exists(csv_path), "summary.csv not created"
        df = pd.read_csv(csv_path)
        assert "ticker" in df.columns
        assert "num_clusters" in df.columns
        print(f"✓ summary.csv valid ({len(df)} rows)")
        print(f"    {df.to_dict('records')}")

        print("\n══════════════════════════════════════")
        print("  ALL GRAPH BUILDER TESTS PASSED ✓")
        print("══════════════════════════════════════")

    finally:
        config.DATA_DIR = orig_data_dir
        config.GRAPH_DIR = orig_graph_dir
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
