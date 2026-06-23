"""
Inference pipeline for the linguistic fingerprinter.

Loads a fine-tuned DistilBERT model and provides:
- Single-batch prediction: ``predict(texts) → list[dict]``
- JSONL pipeline: reads ``data/raw/*.jsonl``, appends ``ai_prob`` and
  ``is_ai_generated``, writes to ``data/analyzed/{ticker}_{date}_bert.jsonl``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

logger = logging.getLogger(__name__)


class LinguisticFingerprinter:
    """Batch inference for AI-generated post detection."""

    def __init__(
        self,
        model_dir: str = str(Path(config.CLASSIFIER_DIR) / "final"),
        threshold: float = config.AI_PROB_THRESHOLD,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_length = max_length

        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}. "
                "Run train_classifier.py first."
            )

        logger.info("Loading fine-tuned model from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )
        self._model.eval()

        # Device selection
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        logger.info("Inference device: %s", self._device)

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Return ``{ai_prob, is_ai_generated}`` for each text.

        Parameters
        ----------
        texts
            List of post texts to classify.

        Returns
        -------
        list[dict]
            One dict per text with keys ``ai_prob`` (float 0–1) and
            ``is_ai_generated`` (bool, True if ai_prob ≥ threshold).
        """
        all_probs: list[float] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encodings = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**encodings).logits

            # Softmax → probability of class 1 (AI)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

        results = []
        for prob in all_probs:
            results.append(
                {
                    "ai_prob": round(prob, 4),
                    "is_ai_generated": prob >= self.threshold,
                }
            )
        return results

    # ── JSONL pipeline ────────────────────────────────────────────

    def process_jsonl(
        self,
        ticker: str,
        data_dir: str = config.DATA_DIR,
        output_dir: str = config.ANALYZED_DIR,
    ) -> Optional[Path]:
        """Process all JSONL files for *ticker*, append AI classification.

        Reads from ``data/raw/{ticker}_*.jsonl``, writes enriched records
        to ``data/analyzed/{ticker}_{date}_bert.jsonl``.
        """
        raw_dir = Path(data_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load all posts for ticker
        posts: list[dict] = []
        for path in sorted(raw_dir.glob(f"{ticker}_*.jsonl")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        posts.append(json.loads(line))

        if not posts:
            logger.info("No posts found for ticker=%s in %s", ticker, raw_dir)
            return None

        logger.info("Classifying %d posts for ticker=%s", len(posts), ticker)

        # Extract texts and predict
        texts = [p.get("text", "") for p in posts]
        predictions = self.predict(texts)

        # Merge predictions into posts
        for post, pred in zip(posts, predictions):
            post["ai_prob"] = pred["ai_prob"]
            post["is_ai_generated"] = pred["is_ai_generated"]

        # Write output
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = out_dir / f"{ticker}_{today}_bert.jsonl"

        with open(out_path, "w", encoding="utf-8") as fh:
            for post in posts:
                fh.write(json.dumps(post, ensure_ascii=False) + "\n")

        # Summary stats
        ai_count = sum(1 for p in predictions if p["is_ai_generated"])
        logger.info(
            "[%s] %d/%d posts flagged as AI-generated (threshold=%.2f) → %s",
            ticker,
            ai_count,
            len(posts),
            self.threshold,
            out_path,
        )
        return out_path

    def process_all_tickers(
        self,
        tickers: Optional[list[str]] = None,
        data_dir: str = config.DATA_DIR,
        output_dir: str = config.ANALYZED_DIR,
    ) -> dict[str, Optional[Path]]:
        """Run the JSONL pipeline for multiple tickers."""
        tickers = tickers or config.TICKERS
        results: dict[str, Optional[Path]] = {}
        for ticker in tickers:
            results[ticker] = self.process_jsonl(ticker, data_dir, output_dir)
        return results
