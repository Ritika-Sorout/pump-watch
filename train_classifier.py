#!/usr/bin/env python3
"""
train_classifier — Fine-tune DistilBERT for AI vs. human post detection.

Workflow:
  1. Generate / load training data (human Reddit posts + AI-generated posts)
  2. Fine-tune distilbert-base-uncased for binary classification
  3. Report F1 and AUC-ROC on the held-out test set
  4. Save the fine-tuned model to models/fingerprinter/final/

Usage
-----
    # Full pipeline: generate data + train
    python train_classifier.py

    # Skip data generation (use cached data)
    python train_classifier.py --skip-datagen

    # Custom hyperparameters
    python train_classifier.py --epochs 5 --batch-size 32 --lr 3e-5

    # Use existing dataset file
    python train_classifier.py --dataset data/training/fingerprinter/dataset.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

import config

LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FMT, level=level, stream=sys.stderr)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for AI-generated post detection."
    )
    p.add_argument(
        "--skip-datagen",
        action="store_true",
        help="Skip data generation, use cached dataset",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Path to existing dataset JSON (skips data generation)",
    )
    p.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.TRAIN_LR)
    p.add_argument("--model", default=config.CLASSIFIER_MODEL)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("train_classifier")

    # ── Step 1: Dataset ───────────────────────────────────────────
    dataset_path: Path
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error("Dataset file not found: %s", dataset_path)
            sys.exit(1)
    elif args.skip_datagen:
        dataset_path = Path("data/training/fingerprinter/dataset.json")
        if not dataset_path.exists():
            logger.error(
                "Cached dataset not found at %s. Remove --skip-datagen.", dataset_path
            )
            sys.exit(1)
    else:
        logger.info("━━━ Step 1: Generating training data ━━━")
        from classifier.data_gen import DatasetGenerator

        gen = DatasetGenerator()
        dataset_path = gen.generate()

    logger.info("Dataset: %s", dataset_path)

    # ── Step 2: Train ─────────────────────────────────────────────
    logger.info("━━━ Step 2: Fine-tuning %s ━━━", args.model)
    from classifier.trainer import FingerprinterTrainer

    trainer = FingerprinterTrainer(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    metrics = trainer.train(dataset_path)

    # ── Summary ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("═" * 60)
    for k, v in metrics.items():
        logger.info("  %-20s %s", k, v)
    logger.info("═" * 60)
    logger.info("  Model saved to: %s/final/", config.CLASSIFIER_DIR)
    logger.info("  Run inference:  python run_classifier.py")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
