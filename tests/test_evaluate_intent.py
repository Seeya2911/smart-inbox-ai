"""Regression tests for intent evaluation model selection."""
from __future__ import annotations

from ml.intent_classifier import EmbeddingIntentClassifier, TfidfIntentClassifier


def test_tfidf_model_is_not_reported_as_transformer() -> None:
    """A TF-IDF artifact must not be mislabeled as a transformer evaluation."""
    tfidf = TfidfIntentClassifier(seed=42)
    transformer = EmbeddingIntentClassifier(seed=42)

    assert tfidf.__class__.__name__ == "TfidfIntentClassifier"
    assert transformer.__class__.__name__ == "EmbeddingIntentClassifier"
    assert not hasattr(tfidf, "model_name")
    assert hasattr(transformer, "model_name")
