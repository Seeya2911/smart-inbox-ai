"""Unit tests for validation error analysis and diagnostic modules."""
import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from ml.error_analysis import (
    analyze_intent_errors,
    analyze_priority_errors,
    compute_calibration_metrics,
    compute_confidence_buckets,
    diagnose_label_conflicts,
    run_error_analysis,
)
from ml.schema import CanonicalEmailExample
from ml.train_multi_output import MultiOutputClassifier


def _make_dummy_example(ex_id: str, subject: str, intent: str, priority: str) -> CanonicalEmailExample:
    namespaced_id = f"synthetic_{ex_id}" if not ex_id.startswith("synthetic_") else ex_id
    return CanonicalEmailExample(
        id=namespaced_id,
        subject=subject,
        body=f"Body for {subject}",
        intent=intent,
        priority=priority,
    )


class TestCalibrationMetrics:
    def test_perfect_calibration(self) -> None:
        y_true = ["low", "low", "high", "high"]
        y_pred = ["low", "low", "high", "high"]
        proba = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        classes = ["low", "high"]

        metrics = compute_calibration_metrics(y_true, y_pred, proba, classes, n_bins=5)
        assert metrics["multiclass_brier_score"] == 0.0
        assert metrics["top_confidence_ece"] == 0.0
        assert len(metrics["reliability_bins"]) == 5

    def test_uncalibrated_case(self) -> None:
        y_true = ["low", "high"]
        y_pred = ["high", "low"]
        proba = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
        ])
        classes = ["low", "high"]

        metrics = compute_calibration_metrics(y_true, y_pred, proba, classes, n_bins=5)
        assert metrics["multiclass_brier_score"] > 1.0
        assert metrics["top_confidence_ece"] > 0.5


class TestConfidenceBucketing:
    def test_bucket_ranges(self) -> None:
        y_true = ["low", "medium", "high", "low"]
        y_pred = ["low", "low", "high", "low"]
        confidences = [0.95, 0.55, 0.75, 0.85]

        buckets = compute_confidence_buckets(y_true, y_pred, confidences)
        assert len(buckets["buckets"]) == 6
        assert buckets["overall_avg_confidence"] > 0.7
        assert buckets["correct_avg_confidence"] > buckets["incorrect_avg_confidence"]


class TestPriorityFailureModes:
    def test_all_failure_modes_recorded(self) -> None:
        examples = [
            _make_dummy_example("1", "Urgent alert", "security", "high"),
            _make_dummy_example("2", "Routine newsletter", "promotion", "low"),
            _make_dummy_example("3", "Follow up tomorrow", "follow_up", "medium"),
        ]
        # Simulate high->low failure
        preds = [
            {"intent": "security", "priority": "low", "intent_confidence": 0.9, "priority_confidence": 0.8},
            {"intent": "promotion", "priority": "low", "intent_confidence": 0.9, "priority_confidence": 0.9},
            {"intent": "follow_up", "priority": "medium", "intent_confidence": 0.8, "priority_confidence": 0.7},
        ]

        pri_analysis = analyze_priority_errors(examples, preds)
        assert pri_analysis["total_validation_priority_errors"] == 1
        assert pri_analysis["modes"]["high->low"]["count"] == 1
        assert pri_analysis["modes"]["high->low"]["percentage_of_true_class"] == 100.0


class TestLabelConflictDiagnostics:
    def test_detects_urgent_low_priority_conflict(self) -> None:
        examples = [
            _make_dummy_example("conf1", "Urgent: security breach detected immediately", "security", "low"),
        ]
        preds = [
            {"intent": "security", "priority": "high", "intent_confidence": 0.9, "priority_confidence": 0.95},
        ]

        conflicts = diagnose_label_conflicts(examples, preds)
        assert conflicts["total_conflicts_flagged"] >= 1
        assert conflicts["conflicts"][0]["id"] == "synthetic_conf1"
        assert conflicts["conflicts"][0]["field"] == "priority"


class TestFullErrorAnalysisPipeline:
    def test_run_error_analysis_produces_files(self, tmp_path: Path) -> None:
        train_examples = [
            _make_dummy_example(f"tr_{i}", f"Invoice {i}", "transactional", "low") for i in range(10)
        ] + [
            _make_dummy_example(f"tr_s_{i}", f"Urgent security {i}", "security", "high") for i in range(10)
        ]
        val_examples = [
            _make_dummy_example(f"val_{i}", f"Invoice receipt {i}", "transactional", "low") for i in range(4)
        ] + [
            _make_dummy_example(f"val_s_{i}", f"Urgent breach {i}", "security", "high") for i in range(4)
        ]

        clf = MultiOutputClassifier(seed=42)
        clf.fit(train_examples)

        report = run_error_analysis(val_examples, clf, tmp_path)
        assert (tmp_path / "classification_error_analysis.json").exists()
        assert (tmp_path / "classification_error_analysis.md").exists()
        assert report["dataset_summary"]["validation_examples"] == 8
