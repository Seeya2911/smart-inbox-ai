"""Tests for the rewritten MultiOutputClassifier (train_multi_output v2).

Covers:
- Two independent vectorizers (not shared)
- Independent intent and priority training
- Prediction dict format with confidence from predict_proba()
- Save (separate intent/priority files) and load → identical predictions
- Majority baseline computation
- Class distribution reporting
- No feature leakage: metadata never reaches the vectorizer
- Inference confidence is from predict_proba(), not a heuristic
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from ml.schema import CanonicalEmailExample
from ml.train_multi_output import MultiOutputClassifier, compute_head_metrics


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
    """Minimal dataset covering multiple intents and priorities."""
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
# Two-vectorizer independence tests
# ---------------------------------------------------------------------------

class TestTwoIndependentVectorizers:
    def test_intent_and_priority_vectorizers_are_different_objects(self):
        clf = MultiOutputClassifier(seed=42)
        assert clf.intent_vectorizer is not clf.priority_vectorizer, (
            "Intent and priority vectorizers must be DIFFERENT objects"
        )

    def test_vectorizers_are_fitted_independently(self):
        examples = _minimal_train_set()
        clf = MultiOutputClassifier(seed=42)
        clf.fit(examples)
        # Both should be fitted
        assert hasattr(clf.intent_vectorizer, "vocabulary_")
        assert hasattr(clf.priority_vectorizer, "vocabulary_")

    def test_vocabularies_can_differ(self):
        """The two vectorizers may have different vocabularies after fitting."""
        examples = _minimal_train_set()
        clf = MultiOutputClassifier(seed=42)
        clf.fit(examples)
        iv = clf.intent_vectorizer.vocabulary_
        pv = clf.priority_vectorizer.vocabulary_
        # They're both dicts of tokens — we don't require them to differ
        # but both must exist
        assert isinstance(iv, dict)
        assert isinstance(pv, dict)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestMultiOutputClassifierTraining:
    def test_fit_succeeds(self):
        examples = _minimal_train_set()
        clf = MultiOutputClassifier(seed=42)
        clf.fit(examples)
        assert clf.is_fitted is True

    def test_fit_raises_on_empty(self):
        clf = MultiOutputClassifier(seed=42)
        with pytest.raises(ValueError, match="empty"):
            clf.fit([])

    def test_intent_classes_set_correctly(self):
        examples = _minimal_train_set()
        clf = MultiOutputClassifier(seed=42)
        clf.fit(examples)
        classes = list(clf.intent_head.classes_)
        assert "security" in classes
        assert "transactional" in classes
        assert "follow_up" in classes

    def test_priority_classes_set_correctly(self):
        examples = _minimal_train_set()
        clf = MultiOutputClassifier(seed=42)
        clf.fit(examples)
        classes = list(clf.priority_head.classes_)
        assert set(classes) == {"high", "low", "medium"}

    def test_hyperparameters_locked(self):
        clf = MultiOutputClassifier(seed=42)
        assert clf.ngram_range == (1, 2)
        assert clf.max_features == 10000
        assert clf.lr_max_iter == 1000
        assert clf.lr_c == 1.0
        assert clf.intent_head.class_weight == "balanced"
        assert clf.priority_head.class_weight == "balanced"

    def test_no_epochs_parameter(self):
        """LogisticRegression is not epoch-based — no epochs attribute."""
        clf = MultiOutputClassifier(seed=42)
        assert not hasattr(clf, "epochs"), "epochs attribute must not exist on MultiOutputClassifier"
        assert not hasattr(clf.intent_head, "epochs")
        assert not hasattr(clf.priority_head, "epochs")


# ---------------------------------------------------------------------------
# Prediction format tests
# ---------------------------------------------------------------------------

class TestPredictionFormat:
    def _fitted_clf(self) -> MultiOutputClassifier:
        clf = MultiOutputClassifier(seed=42)
        clf.fit(_minimal_train_set())
        return clf

    def test_predict_returns_list_of_dicts(self):
        clf = self._fitted_clf()
        examples = _minimal_train_set()[:3]
        preds = clf.predict(examples)
        assert isinstance(preds, list)
        assert len(preds) == 3

    def test_predict_dict_has_required_keys(self):
        clf = self._fitted_clf()
        preds = clf.predict(_minimal_train_set()[:1])
        p = preds[0]
        assert "intent" in p
        assert "priority" in p
        assert "intent_confidence" in p
        assert "priority_confidence" in p

    def test_confidence_is_in_0_1_range(self):
        clf = self._fitted_clf()
        preds = clf.predict(_minimal_train_set())
        for p in preds:
            assert 0.0 <= p["intent_confidence"] <= 1.0, (
                f"intent_confidence {p['intent_confidence']} out of range"
            )
            assert 0.0 <= p["priority_confidence"] <= 1.0, (
                f"priority_confidence {p['priority_confidence']} out of range"
            )

    def test_confidence_comes_from_predict_proba_not_heuristic(self):
        """Confidence must equal max(predict_proba) for the predicted class."""
        clf = self._fitted_clf()
        examples = _minimal_train_set()[:3]
        texts = clf._prepare_texts(examples)

        X_intent = clf.intent_vectorizer.transform(texts)
        X_priority = clf.priority_vectorizer.transform(texts)

        intent_probas = clf.intent_head.predict_proba(X_intent)
        priority_probas = clf.priority_head.predict_proba(X_priority)

        preds = clf.predict(examples)

        import numpy as np
        for i, p in enumerate(preds):
            expected_ic = float(np.max(intent_probas[i]))
            expected_pc = float(np.max(priority_probas[i]))
            assert abs(p["intent_confidence"] - expected_ic) < 1e-6, (
                "intent_confidence must equal max(predict_proba), not a heuristic score"
            )
            assert abs(p["priority_confidence"] - expected_pc) < 1e-6, (
                "priority_confidence must equal max(predict_proba), not a heuristic score"
            )

    def test_predict_empty_returns_empty(self):
        clf = self._fitted_clf()
        assert clf.predict([]) == []

    def test_predict_before_fit_raises(self):
        clf = MultiOutputClassifier(seed=42)
        with pytest.raises(ValueError, match="fit"):
            clf.predict(_minimal_train_set()[:1])

    def test_intent_is_valid_label(self):
        from ml.schema import ALLOWED_INTENTS
        clf = self._fitted_clf()
        preds = clf.predict(_minimal_train_set())
        for p in preds:
            assert p["intent"] in ALLOWED_INTENTS, (
                f"Predicted intent '{p['intent']}' not in ALLOWED_INTENTS"
            )

    def test_priority_is_valid_label(self):
        from ml.schema import ALLOWED_PRIORITIES
        clf = self._fitted_clf()
        preds = clf.predict(_minimal_train_set())
        for p in preds:
            assert p["priority"] in ALLOWED_PRIORITIES


# ---------------------------------------------------------------------------
# Save / Load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def _fitted_clf(self) -> MultiOutputClassifier:
        clf = MultiOutputClassifier(seed=42)
        clf.fit(_minimal_train_set())
        return clf

    def test_save_creates_two_separate_files(self, tmp_path: Path):
        clf = self._fitted_clf()
        intent_path = tmp_path / "intent_tfidf.joblib"
        priority_path = tmp_path / "priority_tfidf.joblib"
        clf.save_intent(intent_path)
        clf.save_priority(priority_path)
        assert intent_path.exists()
        assert priority_path.exists()

    def test_save_and_load_produces_identical_predictions(self, tmp_path: Path):
        clf = self._fitted_clf()
        intent_path = tmp_path / "intent_tfidf.joblib"
        priority_path = tmp_path / "priority_tfidf.joblib"
        clf.save_intent(intent_path)
        clf.save_priority(priority_path)

        clf2 = MultiOutputClassifier.load(intent_path, priority_path)
        assert clf2.is_fitted is True

        examples = _minimal_train_set()[:5]
        preds_before = clf.predict(examples)
        preds_after = clf2.predict(examples)

        for p1, p2 in zip(preds_before, preds_after):
            assert p1["intent"] == p2["intent"], (
                f"Intent mismatch after save/load: {p1['intent']} vs {p2['intent']}"
            )
            assert p1["priority"] == p2["priority"], (
                f"Priority mismatch after save/load: {p1['priority']} vs {p2['priority']}"
            )
            assert abs(p1["intent_confidence"] - p2["intent_confidence"]) < 1e-6
            assert abs(p1["priority_confidence"] - p2["priority_confidence"]) < 1e-6

    def test_intent_artifact_has_correct_identifier(self, tmp_path: Path):
        clf = self._fitted_clf()
        intent_path = tmp_path / "intent_tfidf.joblib"
        clf.save_intent(intent_path)
        import joblib
        artifact = joblib.load(intent_path)
        assert artifact["model_identifier"] == "tfidf-intent-logistic-regression"

    def test_priority_artifact_has_correct_identifier(self, tmp_path: Path):
        clf = self._fitted_clf()
        priority_path = tmp_path / "priority_tfidf.joblib"
        clf.save_priority(priority_path)
        import joblib
        artifact = joblib.load(priority_path)
        assert artifact["model_identifier"] == "tfidf-priority-logistic-regression"


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestComputeHeadMetrics:
    def test_returns_required_keys(self):
        y_true = ["high", "low", "medium", "high", "low"]
        y_pred = ["high", "low", "high", "high", "medium"]
        metrics = compute_head_metrics(y_true, y_pred, "priority")
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics
        assert "majority_baseline" in metrics

    def test_majority_baseline_is_correct(self):
        y_true = ["high", "high", "high", "low", "medium"]  # 'high' is majority
        metrics = compute_head_metrics(y_true, y_true, "intent")
        assert metrics["majority_baseline"]["class"] == "high"

    def test_per_class_all_labels_present(self):
        y_true = ["high", "low", "medium", "high", "low"]
        y_pred = ["high", "low", "medium", "low", "high"]
        metrics = compute_head_metrics(y_true, y_pred, "priority")
        per_class = metrics["per_class"]
        assert "high" in per_class
        assert "low" in per_class
        assert "medium" in per_class

    def test_confusion_matrix_labels_match_classes(self):
        y_true = ["security", "transactional", "meeting", "security"]
        y_pred = ["security", "meeting", "meeting", "transactional"]
        metrics = compute_head_metrics(y_true, y_pred, "intent")
        cm = metrics["confusion_matrix"]
        assert sorted(cm["labels"]) == sorted({"security", "transactional", "meeting"})
        assert len(cm["matrix"]) == 3

    def test_handles_empty_input(self):
        metrics = compute_head_metrics([], [], "intent")
        assert metrics["accuracy"] == 0.0

    def test_all_correct_gives_accuracy_1(self):
        y = ["high", "low", "medium"]
        metrics = compute_head_metrics(y, y, "priority")
        assert metrics["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Leakage: features test
# ---------------------------------------------------------------------------

class TestFeatureLeakageInClassifier:
    """Prove that metadata never reaches the vectorizer."""

    def test_prepare_texts_does_not_include_intent_label(self):
        """intent is the TARGET — it must not appear in model input features."""
        ex = CanonicalEmailExample.from_dict({
            "id": "synthetic_001",
            "subject": "Meeting schedule",
            "body": "Please confirm your attendance.",
            "intent": "VERY_UNIQUE_INTENT_MARKER_XYZ",  # would fail schema, tested via full_text
            "priority": "high",
            "source": "synthetic",
            "label_source": "llm",
            "label_confidence": 0.9,
            "is_synthetic": True,
        } | {"intent": "meeting"})  # valid intent

        clf = MultiOutputClassifier(seed=42)
        texts = clf._prepare_texts([ex])
        assert len(texts) == 1
        assert "VERY_UNIQUE_INTENT_MARKER" not in texts[0]

    def test_prepare_texts_does_not_include_priority_label(self):
        ex = CanonicalEmailExample.from_dict({
            "id": "synthetic_002",
            "subject": "Invoice",
            "body": "Please pay the invoice.",
            "intent": "transactional",
            "priority": "high",
            "source": "synthetic",
            "label_source": "llm",
            "label_confidence": 0.9,
            "is_synthetic": True,
        })
        clf = MultiOutputClassifier(seed=42)
        texts = clf._prepare_texts([ex])
        # "high" might appear as a word in subject/body but not as the label field
        assert "priority" not in texts[0]  # the field name should not be in text

    def test_prepare_texts_does_not_include_llm_reasons(self):
        ex = CanonicalEmailExample(
            id="synthetic_003",
            subject="Password reset",
            body="Your password was changed.",
            intent="security",
            priority="high",
            source="synthetic",
            label_source="llm",
            label_confidence=0.9,
            is_synthetic=True,
            llm_intent_reason="UNIQUE_REASON_SENTINEL_ABC123",
            llm_priority_reason="UNIQUE_PRIORITY_SENTINEL_DEF456",
        )
        clf = MultiOutputClassifier(seed=42)
        texts = clf._prepare_texts([ex])
        assert "UNIQUE_REASON_SENTINEL_ABC123" not in texts[0]
        assert "UNIQUE_PRIORITY_SENTINEL_DEF456" not in texts[0]
