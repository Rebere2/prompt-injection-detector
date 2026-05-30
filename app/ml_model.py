# ===========================================
# Prompt Injection Detector — ML Model
# ===========================================

"""
Machine learning classifier for prompt injection detection.

Uses a scikit-learn pipeline with TF-IDF features + custom
numeric features, fed into a LogisticRegression classifier.
Falls back gracefully if the model file is not found.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)

# Default model path
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "classifier.pkl"


class NumericFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract numeric features from text for injection detection.

    Features:
        - text_length: character count
        - word_count: number of words
        - uppercase_ratio: proportion of uppercase characters
        - special_char_ratio: proportion of non-alphanumeric characters
        - tag_count: number of markup-like tags
        - separator_count: number of separator patterns
        - question_mark_count: number of question marks
        - exclamation_count: number of exclamation marks
    """

    def fit(self, X: list[str], y: Any = None) -> "NumericFeatureExtractor":
        """Fit is a no-op — this is a stateless transformer."""
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        """Extract numeric features from each text."""
        features = []
        for text in X:
            alpha_chars = [c for c in text if c.isalpha()]
            upper_count = sum(1 for c in alpha_chars if c.isupper())
            alpha_len = len(alpha_chars) if alpha_chars else 1

            features.append([
                len(text),
                len(text.split()),
                upper_count / alpha_len,
                sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
                len(re.findall(r"<[^>]+>|\[[A-Z]+\]", text)),
                len(re.findall(r"[-=*_]{5,}", text)),
                text.count("?"),
                text.count("!"),
            ])

        return np.array(features, dtype=np.float64)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Return feature names for pipeline introspection."""
        return [
            "text_length",
            "word_count",
            "uppercase_ratio",
            "special_char_ratio",
            "tag_count",
            "separator_count",
            "question_mark_count",
            "exclamation_count",
        ]


def build_pipeline() -> Pipeline:
    """
    Build the ML pipeline for injection detection.

    Combines TF-IDF text features with custom numeric features,
    then feeds into a LogisticRegression classifier.

    Returns:
        A scikit-learn Pipeline ready to be fit.
    """
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        analyzer="word",
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    char_tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(2, 5),
        analyzer="char_wb",
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", tfidf),
            ("char_tfidf", char_tfidf),
            ("numeric", NumericFeatureExtractor()),
        ])),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    return pipeline


def load_dataset(data_dir: str | Path | None = None) -> tuple[list[str], list[int]]:
    """
    Load the training dataset from JSONL files.

    Args:
        data_dir: Path to the data directory. Defaults to project data/.

    Returns:
        Tuple of (texts, labels) where 1 = injection, 0 = legitimate.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    texts: list[str] = []
    labels: list[int] = []

    # Load injections (label = 1)
    injections_path = data_dir / "injections.jsonl"
    if injections_path.exists():
        with open(injections_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    texts.append(record["text"])
                    labels.append(1)
        logger.info("Loaded %d injection samples from %s", labels.count(1), injections_path)

    # Load legitimate (label = 0)
    legitimate_path = data_dir / "legitimate.jsonl"
    if legitimate_path.exists():
        with open(legitimate_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    texts.append(record["text"])
                    labels.append(0)
        logger.info(
            "Loaded %d legitimate samples from %s",
            len(labels) - labels.count(1),
            legitimate_path,
        )

    return texts, labels


def train_model(
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Pipeline:
    """
    Train the ML model and serialize it.

    Args:
        data_dir: Path to the data directory.
        output_path: Where to save the model. Defaults to models/classifier.pkl.

    Returns:
        The trained pipeline.
    """
    if output_path is None:
        output_path = MODEL_PATH

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    texts, labels = load_dataset(data_dir)

    if len(texts) < 10:
        raise ValueError(
            f"Need at least 10 samples to train, got {len(texts)}. "
            "Add more data to data/injections.jsonl and data/legitimate.jsonl."
        )

    logger.info("Training model on %d samples...", len(texts))

    pipeline = build_pipeline()
    pipeline.fit(texts, labels)

    joblib.dump(pipeline, output_path)
    logger.info("Model saved to %s", output_path)

    return pipeline


class MLClassifier:
    """
    Wrapper around the trained ML model for inference.

    Loads the model lazily and provides a predict method.
    If the model file doesn't exist, returns None (heuristic-only mode).
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else MODEL_PATH
        self._pipeline: Pipeline | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the ML model is loaded and ready."""
        if not self._loaded:
            self._try_load()
        return self._pipeline is not None

    def _try_load(self) -> None:
        """Attempt to load the model from disk."""
        self._loaded = True
        if self._model_path.exists():
            try:
                self._pipeline = joblib.load(self._model_path)
                logger.info("ML model loaded from %s", self._model_path)
            except Exception as exc:
                logger.warning("Failed to load ML model: %s", exc)
                self._pipeline = None
        else:
            logger.info(
                "ML model not found at %s — running in heuristic-only mode",
                self._model_path,
            )

    def predict(self, text: str) -> float | None:
        """
        Predict the injection probability for a given text.

        Args:
            text: The input text to classify.

        Returns:
            A float between 0.0 and 1.0 representing injection probability,
            or None if the model is not available.
        """
        if not self.is_loaded:
            return None

        assert self._pipeline is not None
        probabilities = self._pipeline.predict_proba([text])
        # Column 1 is the injection class probability
        injection_prob = float(probabilities[0][1])
        return round(injection_prob, 4)
