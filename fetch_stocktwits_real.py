"""
fetch_stocktwits_real.py — pull real StockTwits data into a CSV
=================================================================
Hits StockTwits' public, unauthenticated read endpoint:

    https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json

No API key needed. Anonymous access is rate-limited (roughly 200 requests/
hour per IP) and each call returns the ~30 most recent messages for that
ticker — so this gets you real, current data, not a big historical archive.
(For bulk historical StockTwits data — e.g. 2020-2022 — see the "Kaggle
datasets" note at the bottom of this file instead.)

Usage:
    pip install requests pandas
    python fetch_stocktwits_real.py --tickers GME AMC TSLA NVDA NOK

Output:
    data/raw/<TICKER>_stocktwits.csv    one row per message
    data/raw/<TICKER>_stocktwits.jsonl  same data, JSONL — matches the
                                        schema your other crawlers
                                        (reddit.py, twitter.py) already write,
                                        so build_graph.py can ingest it as-is.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import requests

API_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
OUT_DIR = Path("data/raw")

FIELDNAMES = [
    "post_id", "ticker", "platform", "author", "author_followers",
    "timestamp", "content", "sentiment", "likes",
]


def fetch_ticker(ticker: str) -> list[dict]:
    """Hit the public stream endpoint for one ticker and return parsed rows."""
    url = API_URL.format(ticker=ticker)
    resp = requests.get(url, headers={"User-Agent": "pumpwatch-research/1.0"}, timeout=15)

    if resp.status_code == 429:
        print(f"  [{ticker}] rate-limited (429) — back off and retry later")
        return []
    resp.raise_for_status()

    data = resp.json()
    messages = data.get("messages", [])
    rows = []
    for m in messages:
        entities = m.get("entities") or {}
        sentiment_block = entities.get("sentiment") or {}
        user = m.get("user") or {}
        rows.append({
            "post_id": m.get("id"),
            "ticker": ticker,
            "platform": "stocktwits",
            "author": user.get("username"),
            "author_followers": user.get("followers", 0),
            "timestamp": m.get("created_at"),       # ISO 8601, e.g. 2026-06-26T14:03:11Z
            "content": (m.get("body") or "").replace("\n", " ").strip(),
            "sentiment": sentiment_block.get("basic"),  # "Bullish" / "Bearish" / None
            "likes": (m.get("likes") or {}).get("total", 0),
        })
    return rows


def save_csv(rows: list[dict], ticker: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{ticker}_stocktwits.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> wrote {len(rows)} rows to {path}")
    return path


def save_jsonl(rows: list[dict], ticker: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{ticker}_stocktwits.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  -> wrote {len(rows)} rows to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Pull real StockTwits messages into CSV/JSONL")
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g. --tickers GME AMC TSLA")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests (be polite to the rate limit)")
    args = parser.parse_args()

    for ticker in args.tickers:
        print(f"Fetching {ticker} ...")
        try:
            rows = fetch_ticker(ticker)
        except requests.RequestException as e:
            print(f"  [{ticker}] request failed: {e}")
            continue

        if not rows:
            print(f"  [{ticker}] no messages returned")
            continue

        save_csv(rows, ticker)
        save_jsonl(rows, ticker)
        time.sleep(args.delay)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Need MUCH more historical volume than ~30 recent messages per ticker?
# Real, larger StockTwits datasets exist publicly on Kaggle — download via
# the Kaggle CLI (`pip install kaggle`, needs a free kaggle.com account +
# API token in ~/.kaggle/kaggle.json), then run a small column-rename pass
# to match FIELDNAMES above before dropping the CSV into data/raw/:
#
#   kaggle datasets download -d frankcaoyun/stocktwits-2020-2022-raw
#   # covers AAPL, AMZN, FB(META), NVDA, TSLA, Jan 2020 - Mar 2022
#
#   kaggle datasets download -d rutviknelluri/tweets-of-indian-stocks-from-stocktwits
#   # Indian-market tickers, if relevant to your TICKERS list in config.py
#
# Ask me to write the column-mapping script once you've picked one and can
# show me its actual header row — Kaggle uploads vary in column names.
# ---------------------------------------------------------------------------
