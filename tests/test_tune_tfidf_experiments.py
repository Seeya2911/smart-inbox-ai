"""Unit tests for tunable TF-IDF multi-output classifier and experiment runner."""
from pathlib import Path

import pytest

from ml.schema import CanonicalEmailExample
from ml.tune_tfidf_experiments import TunableMultiOutputClassifier, run_tuning_grid


def _make_dummy_example(ex_id: str, text: str, intent: str, priority: str) -> CanonicalEmailExample:
    namespaced_id = f"synthetic_{ex_id}" if not ex_id.startswith("synthetic_") else ex_id
    return CanonicalEmailExample(
        id=namespaced_id,
        subject=text,
        body=f"Details for {text}",
        intent=intent,
        priority=priority,
    )


class TestTunableClassifier:
    def test_fit_predict_save_load(self, tmp_path: Path) -> None:
        train = [
            _make_dummy_example(f"tr_{i}", f"Action required please complete task {i}", "request", "high")
            for i in range(8)
        ] + [
            _make_dummy_example(f"tr_l_{i}", f"Weekly newsletter and updates {i}", "information", "low")
            for i in range(8)
        ]
        val = [
            _make_dummy_example(f"val_{i}", f"Action task {i}", "request", "high") for i in range(2)
        ] + [
            _make_dummy_example(f"val_l_{i}", f"Newsletter {i}", "information", "low") for i in range(2)
        ]

        clf = TunableMultiOutputClassifier(ngram_range=(1, 2), min_df=1, c_intent=1.0, c_priority=1.0)
        clf.fit(train)

        preds = clf.predict(val)
        assert len(preds) == 4
        assert "intent" in preds[0]
        assert "priority" in preds[0]
        assert "intent_confidence" in preds[0]
        assert "priority_confidence" in preds[0]

        # Save and verify
        intent_art = tmp_path / "intent_test.joblib"
        pri_art = tmp_path / "priority_test.joblib"
        clf.save(intent_art, pri_art)
        assert intent_art.exists()
        assert pri_art.exists()

    def test_tuning_grid_execution(self, tmp_path: Path) -> None:
        train = [
            _make_dummy_example(f"tr_{i}", f"Meeting call schedule {i}", "meeting", "medium") for i in range(6)
        ] + [
            _make_dummy_example(f"tr_s_{i}", f"Password reset urgent {i}", "security", "high") for i in range(6)
        ]
        val = [
            _make_dummy_example(f"val_{i}", f"Meeting tomorrow {i}", "meeting", "medium") for i in range(2)
        ] + [
            _make_dummy_example(f"val_s_{i}", f"Urgent security {i}", "security", "high") for i in range(2)
        ]
        test = val

        result = run_tuning_grid(train, val, test, tmp_path, seed=42)
        assert "experiments_summary" in result
        assert "selected_best_configuration" in result
        assert "decision" in result
        assert (tmp_path / "experiments" / "classification_comparison.json").exists()
