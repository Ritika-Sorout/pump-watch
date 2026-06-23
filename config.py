"""
Crawler configuration.

Edit TICKERS to control which symbols are crawled.
All other settings have sensible defaults.
"""

# ── Ticker symbols to track ──────────────────────────────────────────
TICKERS: list[str] = ["GME", "AMC", "DOGE"]

# ── Reddit ────────────────────────────────────────────────────────────
REDDIT_SUBREDDITS: list[str] = ["wallstreetbets", "stocks", "investing"]

# ── Crawl limits ──────────────────────────────────────────────────────
MAX_POSTS_PER_SOURCE: int = 100   # per ticker, per platform, per run

# ── Rate-limit / backoff ──────────────────────────────────────────────
BACKOFF_BASE_SECONDS: float = 2.0
BACKOFF_MAX_SECONDS: float = 64.0
BACKOFF_JITTER_FACTOR: float = 0.25   # ±25 %

# ── Output ────────────────────────────────────────────────────────────
DATA_DIR: str = "data/raw"
DEDUP_DIR: str = "data"
GRAPH_DIR: str = "data/graphs"

# ── Graph builder ─────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.85
TIME_WINDOW_MINUTES: int = 15
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = 256

# ── BERT classifier ──────────────────────────────────────────────────
CLASSIFIER_MODEL: str = "distilbert-base-uncased"
CLASSIFIER_DIR: str = "models/fingerprinter"   # saved fine-tuned model
ANALYZED_DIR: str = "data/analyzed"
AI_PROB_THRESHOLD: float = 0.7
TRAIN_EPOCHS: int = 3
TRAIN_BATCH_SIZE: int = 16
TRAIN_LR: float = 2e-5
TRAIN_EVAL_SPLIT: float = 0.2                  # hold-out fraction
DATAGEN_TICKERS: list[str] = [
    "GME", "AMC", "TSLA", "PLTR", "BB", "NOK", "AAPL", "MSFT",
    "NVDA", "DOGE", "SPY", "QQQ", "SOFI", "RIVN", "NIO",
]
DATAGEN_AI_SAMPLES_PER_TICKER: int = 10
DATAGEN_HUMAN_POSTS_PER_SUBREDDIT: int = 200

