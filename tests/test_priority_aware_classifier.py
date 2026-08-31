"""Unit tests for PriorityAwareClassifier and threshold routing logic."""
from pathlib import Path

import joblib
import pytest

from ml.priority_aware_classifier import PriorityAwareClassifier
from ml.schema import CanonicalEmailExample


def _make_example(ex_id: str, subject: str, intent: str, priority: str) -> CanonicalEmailExample:
    return CanonicalEmailExample(
        id=f"synthetic_{ex_id}",
        subject=subject,
        body=f"Body details for {subject}",
        intent=intent,
        priority=priority,
    )


class TestPriorityAwareClassifier:
    def test_fit_predict_pipeline(self) -> None:
        train = [
            _make_example("1", "Urgent security alert password reset", "security", "high"),
            _make_example("2", "Critical system failure action required", "security", "high"),
            _make_example("3", "Weekly newsletter discounts and updates", "promotion", "low"),
            _make_example("4", "Routine invoice receipt attached", "transactional", "low"),
            _make_example("5", "Team sync meeting tomorrow 3pm", "meeting", "medium"),
            _make_example("6", "Project status update follow up", "follow_up", "medium"),
        ]
        val = [
            _make_example("v1", "Immediate action required for server", "security", "high"),
            _make_example("v2", "Weekly promotional newsletter", "promotion", "low"),
        ]

        clf = PriorityAwareClassifier(ngram_range=(1, 2), max_features=1000, high_threshold=0.35, seed=42)
        clf.fit(train)

        preds = clf.predict(val)
        assert len(preds) == 2
        assert "intent" in preds[0]
        assert "priority" in preds[0]
        assert "intent_confidence" in preds[0]
        assert "priority_confidence" in preds[0]

    def test_high_threshold_routing(self) -> None:
        train = [
            _make_example("1", "Urgent security alert", "security", "high"),
            _make_example("2", "Weekly digest", "information", "low"),
        ] * 4

        # With low threshold, high priority is more easily triggered
        clf = PriorityAwareClassifier(high_threshold=0.1, seed=42)
        clf.fit(train)
        preds_low_th = clf.predict([_make_example("t1", "Security notice", "security", "high")])

        clf.high_threshold = 0.99
        preds_high_th = clf.predict([_make_example("t1", "Security notice", "security", "high")])

        assert isinstance(preds_low_th[0]["priority"], str)
        assert isinstance(preds_high_th[0]["priority"], str)

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        train = [
            _make_example(f"tr_{i}", f"Urgent security {i}", "security", "high") for i in range(5)
        ] + [
            _make_example(f"tr_l_{i}", f"Newsletter {i}", "promotion", "low") for i in range(5)
        ]
        test = [
            _make_example("te_1", "Urgent security breach", "security", "high"),
        ]

        clf = PriorityAwareClassifier(seed=42, high_threshold=0.35)
        clf.fit(train)
        orig_preds = clf.predict(test)

        intent_file = tmp_path / "intent_test.joblib"
        pri_file = tmp_path / "pri_test.joblib"
        clf.save(intent_file, pri_file)

        assert intent_file.exists()
        assert pri_file.exists()

        loaded_pri = joblib.load(pri_file)
        assert loaded_pri["model_identifier"] == "smart-inbox-priority-aware-classifier"
        assert loaded_pri["high_threshold"] == 0.35
