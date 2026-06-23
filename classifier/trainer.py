"""
DistilBERT fine-tuning trainer for the linguistic fingerprinter.

Wraps HuggingFace ``Trainer`` to fine-tune ``distilbert-base-uncased``
on a binary classification task:
    label 0 = human-written financial post
    label 1 = AI-generated financial post

Reports F1 (macro) and AUC-ROC on the held-out evaluation set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

import config

logger = logging.getLogger(__name__)


def _compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    """Compute F1 and AUC-ROC for the eval set."""
    logits = pred.predictions
    labels = pred.label_ids

    # Softmax → probabilities for class 1
    probs = _softmax(logits)[:, 1]
    preds = (probs >= config.AI_PROB_THRESHOLD).astype(int)

    f1 = f1_score(labels, preds, average="macro")

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # Only one class present in y_true
        auc = 0.0

    return {"f1_macro": round(f1, 4), "auc_roc": round(auc, 4)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class FingerprinterTrainer:
    """Fine-tune DistilBERT for AI vs. human post classification."""

    def __init__(
        self,
        model_name: str = config.CLASSIFIER_MODEL,
        output_dir: str = config.CLASSIFIER_DIR,
        epochs: int = config.TRAIN_EPOCHS,
        batch_size: int = config.TRAIN_BATCH_SIZE,
        lr: float = config.TRAIN_LR,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "human", 1: "ai"},
            label2id={"human": 0, "ai": 1},
        )

    # ── Public API ────────────────────────────────────────────────

    def train(
        self,
        dataset_path: str | Path,
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fine-tune the model and return evaluation metrics.

        Parameters
        ----------
        dataset_path
            Path to the JSON file produced by ``DatasetGenerator.generate()``.
        resume_from_checkpoint
            Optional path to a checkpoint to resume training from.

        Returns
        -------
        dict
            Evaluation metrics including ``f1_macro`` and ``auc_roc``.
        """
        ds = self._load_dataset(dataset_path)

        logger.info(
            "Training: model=%s  epochs=%d  batch=%d  lr=%s",
            self.model_name,
            self.epochs,
            self.batch_size,
            self.lr,
        )
        logger.info("  train=%d  eval=%d", len(ds["train"]), len(ds["test"]))

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            fp16=False,  # MPS/CPU friendly
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            processing_class=self._tokenizer,
            compute_metrics=_compute_metrics,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Final evaluation
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)

        # Save model + tokenizer
        save_path = Path(self.output_dir) / "final"
        trainer.save_model(str(save_path))
        self._tokenizer.save_pretrained(str(save_path))
        logger.info("Saved fine-tuned model → %s", save_path)

        # Save metrics
        metrics_path = Path(self.output_dir) / "eval_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

        return metrics

    # ── Internals ─────────────────────────────────────────────────

    def _load_dataset(self, path: str | Path) -> DatasetDict:
        """Load the JSON dataset and tokenize it."""
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        ds = DatasetDict(
            {
                "train": Dataset.from_list(raw["train"]),
                "test": Dataset.from_list(raw["test"]),
            }
        )

        def tokenize(batch):
            return self._tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        ds = ds.map(tokenize, batched=True, remove_columns=["text"])
        ds.set_format("torch")
        return ds
