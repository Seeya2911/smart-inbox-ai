"""Tests for ml.train_multi_output_embedding -- Sentence Embeddings Multi-Output Classifier.

Covers:
- No feature leakage: metadata never reaches encoder or classifier
- Independent intent and priority heads
- Deterministic Logistic Regression configuration
- Prediction output format with confidence scores
- Save/load verification
- Comparison report generation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from ml.schema import ALLOWED_INTENTS, ALLOWED_PRIORITIES, CanonicalEmailExample
from ml.train_multi_output_embedding import (
    EmbeddingMultiOutputClassifier,
    build_comparison_report,
    measure_inference_latency,
)


# ---------------------------------------------------------------------------
# Mock Encoder for fast, deterministic unit testing
# ---------------------------------------------------------------------------

class MockSentenceEncoder:
    """Deterministic mock encoder that outputs fixed-dim embeddings based on text hash."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Deterministic pseudo-embedding based on character counts
            vec = np.zeros(self.dim, dtype=np.float32)
            for idx, ch in enumerate(text[: self.dim]):
                vec[idx % self.dim] += (ord(ch) % 17) + 1.0
            if normalize_embeddings and np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ex(idx: int, intent: str, priority: str, extra: str = "") -> CanonicalEmailExample:
    return CanonicalEmailExample.from_dict({
        "id": f"synthetic_{idx:04d}",
        "subject": f"Subject {idx} {intent}",
        "body": f"Email body {idx} discussing {intent} issues. {extra} "
                f"This message requires attention for {priority} priority matters.",
        "intent": intent,
        "priority": priority,
        "source": "synthetic",
        "label_source": "llm",
        "label_confidence": 0.9,
        "is_synthetic": True,
    })


def _minimal_train_set() -> List[CanonicalEmailExample]:
    examples = []
    idx = 0
    combos = [
        ("security", "high"), ("transactional", "low"), ("meeting", "medium"),
        ("request", "high"), ("information", "low"), ("question", "medium"),
        ("notification", "low"), ("promotion", "low"), ("complaint", "high"),
        ("follow_up", "medium"), ("other", "low"),
    ]
    for intent, priority in combos:
        for k in range(3):
            idx += 1
            examples.append(_make_ex(idx, intent, priority, f"unique_{k}_{intent}"))
    return examples


# ---------------------------------------------------------------------------
# Feature Leakage Tests
# ---------------------------------------------------------------------------

class TestEmbeddingFeatureLeakage:
    """Strictly prove metadata fields never reach the embedding model."""

    def test_prepare_texts_does_not_contain_intent_reason(self):
        ex = CanonicalEmailExample(
            id="synthetic_001",
            subject="Security Incident",
            body="Suspicious login detected.",
            intent="security",
            priority="high",
            source="synthetic",
            label_source="llm",
            label_confidence=0.9,
            is_synthetic=True,
            llm_intent_reason="SENTINEL_INTENT_REASON_XYZ_123",
            llm_priority_reason="SENTINEL_PRIORITY_REASON_ABC_456",
        )
        clf = EmbeddingMultiOutputClassifier(embedding_model=MockSentenceEncoder())
        texts = clf._prepare_texts([ex])
        assert len(texts) == 1
        assert "SENTINEL_INTENT_REASON_XYZ_123" not in texts[0]
        assert "SENTINEL_PRIORITY_REASON_ABC_456" not in texts[0]

    def test_prepare_texts_does_not_contain_metadata_fields(self):
        ex = _make_ex(1, "request", "high")
        clf = EmbeddingMultiOutputClassifier(embedding_model=MockSentenceEncoder())
        texts = clf._prepare_texts([ex])
        assert "label_confidence" not in texts[0]
        assert "label_source" not in texts[0]
        assert "is_synthetic" not in texts[0]


# ---------------------------------------------------------------------------
# Classifier Training & Architecture Tests
# ---------------------------------------------------------------------------

class TestEmbeddingClassifier:

    def test_independent_heads(self):
        clf = EmbeddingMultiOutputClassifier(embedding_model=MockSentenceEncoder())
        assert clf.intent_head is not clf.priority_head
        assert clf.intent_head.class_weight == "balanced"
        assert clf.priority_head.class_weight == "balanced"
        assert clf.intent_head.max_iter == 1000
        assert clf.priority_head.max_iter == 1000

    def test_fit_and_predict(self):
        examples = _minimal_train_set()
        clf = EmbeddingMultiOutputClassifier(embedding_model=MockSentenceEncoder(dim=32), seed=42)
        clf.fit(examples)
        assert clf.is_fitted is True
        assert clf.embedding_dim == 32

        preds = clf.predict(examples[:3])
        assert len(preds) == 3
        for p in preds:
            assert "intent" in p
            assert "priority" in p
            assert 0.0 <= p["intent_confidence"] <= 1.0
            assert 0.0 <= p["priority_confidence"] <= 1.0
            assert p["intent"] in ALLOWED_INTENTS
            assert p["priority"] in ALLOWED_PRIORITIES

    def test_save_and_load(self, tmp_path: Path):
        examples = _minimal_train_set()
        mock_enc = MockSentenceEncoder(dim=32)
        clf = EmbeddingMultiOutputClassifier(embedding_model=mock_enc, seed=42)
        clf.fit(examples)

        intent_path = tmp_path / "intent_embedding.joblib"
        priority_path = tmp_path / "priority_embedding.joblib"
        clf.save_intent(intent_path)
        clf.save_priority(priority_path)

        assert intent_path.exists()
        assert priority_path.exists()

        # Load back
        clf2 = EmbeddingMultiOutputClassifier.load(
            intent_path, priority_path, embedding_model=mock_enc
        )
        assert clf2.is_fitted is True

        preds1 = clf.predict(examples[:5])
        preds2 = clf2.predict(examples[:5])

        for p1, p2 in zip(preds1, preds2):
            assert p1["intent"] == p2["intent"]
            assert p1["priority"] == p2["priority"]
            assert abs(p1["intent_confidence"] - p2["intent_confidence"]) < 1e-6
            assert abs(p1["priority_confidence"] - p2["priority_confidence"]) < 1e-6


# ---------------------------------------------------------------------------
# Comparison Report Helper Tests
# ---------------------------------------------------------------------------

class TestComparisonReport:

    def test_build_comparison_report(self):
        base_intent = {
            "validation": {"macro_f1": 0.70},
            "test": {"accuracy": 0.80, "macro_f1": 0.69, "weighted_f1": 0.80, "per_class": {"security": {"f1": 0.95}}},
        }
        base_pri = {
            "validation": {"macro_f1": 0.72},
            "test": {"accuracy": 0.82, "macro_f1": 0.66, "weighted_f1": 0.83, "per_class": {"high": {"precision": 0.41, "recall": 0.52, "f1": 0.46}}},
        }
        emb_intent = {
            "validation": {"macro_f1": 0.75},
            "test": {"accuracy": 0.83, "macro_f1": 0.74, "weighted_f1": 0.83, "per_class": {"security": {"f1": 0.96}}},
        }
        emb_pri = {
            "validation": {"macro_f1": 0.76},
            "test": {"accuracy": 0.84, "macro_f1": 0.70, "weighted_f1": 0.85, "per_class": {"high": {"precision": 0.50, "recall": 0.60, "f1": 0.55}}},
        }

        report = build_comparison_report(
            baseline_intent_eval=base_intent,
            baseline_priority_eval=base_pri,
            embedding_intent_eval=emb_intent,
            embedding_priority_eval=emb_pri,
        )

        assert "validation_selection" in report
        assert "test_comparison" in report
        assert "per_class_priority" in report
        assert report["recommendation"] == "A"
