#!/usr/bin/env python3
# ===========================================
# Prompt Injection Detector — Model Training
# ===========================================

"""
Standalone training script.

Loads the dataset, trains the ML classifier, evaluates it,
and serializes the model to models/classifier.pkl.

Usage:
    python train.py
    python train.py --data-dir ./data --output ./models/classifier.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

# Add project root to path so we can import from app/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ml_model import build_pipeline, load_dataset, MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train the prompt injection classifier."""
    parser = argparse.ArgumentParser(
        description="Train the prompt injection detection ML model."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the trained model (default: ./models/classifier.pkl)",
    )
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent / "data"
    output_path = Path(args.output) if args.output else MODEL_DIR / "classifier.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset from %s...", data_dir)
    texts, labels = load_dataset(data_dir)

    if len(texts) < 10:
        logger.error(
            "Not enough samples (%d). Need at least 10. "
            "Add more data to data/injections.jsonl and data/legitimate.jsonl.",
            len(texts),
        )
        sys.exit(1)

    labels_arr = np.array(labels)
    n_injection = int(labels_arr.sum())
    n_legitimate = len(labels_arr) - n_injection

    logger.info(
        "Dataset: %d samples (%d injections, %d legitimate)",
        len(texts), n_injection, n_legitimate,
    )

    # Build and train pipeline
    logger.info("Building pipeline...")
    pipeline = build_pipeline()

    # Cross-validation
    logger.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
    logger.info(
        "CV Accuracy: %.4f (+/- %.4f)",
        cv_scores.mean(), cv_scores.std() * 2,
    )

    # Train on full dataset
    logger.info("Training on full dataset...")
    pipeline.fit(texts, labels)

    # Evaluate on training set (for reference)
    predictions = pipeline.predict(texts)
    accuracy = accuracy_score(labels, predictions)

    logger.info("Training accuracy: %.4f", accuracy)
    logger.info("\nClassification Report:\n%s", classification_report(
        labels, predictions,
        target_names=["legitimate", "injection"],
    ))
    logger.info("\nConfusion Matrix:\n%s", confusion_matrix(labels, predictions))

    # Save model
    import joblib
    joblib.dump(pipeline, output_path)
    logger.info("Model saved to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
