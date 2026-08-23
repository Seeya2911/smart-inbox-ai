"""Offline contract tests for the embedding intent classifier.

These tests deliberately inject a tiny deterministic encoder so the classifier
contract can be exercised without downloading a pretrained model. They do not
claim benchmark performance for the real multilingual encoder.
"""
from __future__ import annotations

import numpy as np

from ml.intent_classifier import EmbeddingIntentClassifier, PRETRAINED_MODEL_ID
from ml.schema import CanonicalIntentExample


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        del normalize_embeddings, show_progress_bar
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    1.0 if "send" in lowered or "please" in lowered else 0.0,
                    1.0 if "report" in lowered or "document" in lowered else 0.0,
                    float(len(lowered) % 7) / 7.0,
                ]
            )
        return np.asarray(rows, dtype=float)


def _examples():
    return [
        CanonicalIntentExample("Please send the report", "en", "request", "test", "1", "request"),
        CanonicalIntentExample("Please send the document", "en", "request", "test", "2", "request"),
        CanonicalIntentExample("Here is the report", "en", "information", "test", "3", "information"),
        CanonicalIntentExample("The document is attached", "en", "information", "test", "4", "information"),
    ]


def test_embedding_classifier_uses_declared_pretrained_model_by_default():
    clf = EmbeddingIntentClassifier(encoder_instance=FakeEncoder(), seed=42)
    assert clf.model_name == PRETRAINED_MODEL_ID


def test_embedding_classifier_fit_predict_and_probabilities_offline():
    examples = _examples()
    clf = EmbeddingIntentClassifier(encoder_instance=FakeEncoder(), seed=42)
    clf.fit(examples)

    assert sorted(clf.classes_.tolist()) == ["information", "request"]

    predictions = clf.predict(examples)
    probabilities = clf.predict_proba(examples)

    assert len(predictions) == len(examples)
    assert len(probabilities) == len(examples)
    assert all(set(row) == {"information", "request"} for row in probabilities)
    assert all(abs(sum(row.values()) - 1.0) < 1e-9 for row in probabilities)


def test_embedding_classifier_does_not_change_classes_during_prediction():
    examples = _examples()
    clf = EmbeddingIntentClassifier(encoder_instance=FakeEncoder(), seed=42)
    clf.fit(examples[:2] + examples[2:])
    before = clf.classes_.tolist()

    clf.predict([CanonicalIntentExample("Schedule a meeting", "en", "request", "test", "5", "request")])

    assert clf.classes_.tolist() == before
