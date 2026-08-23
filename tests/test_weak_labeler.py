from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.intent_rules import IntentRuleEngine
from ml.weak_labeler import DualWeakLabeler, process_dataset_file


def test_intent_rule_engine_is_importable_and_returns_weak_score() -> None:
    engine = IntentRuleEngine()
    intent, score, reasons = engine.predict_weak_intent(
        "Password reset required", "Please reset your password after this security alert."
    )
    assert intent == "security"
    assert score > 0
    assert reasons


def test_rules_do_not_claim_calibrated_confidence() -> None:
    labeler = DualWeakLabeler()
    example = labeler.evaluate_email(
        {
            "id": "enron_1",
            "source": "enron",
            "subject": "Password reset required",
            "body": "Please reset your password after this security alert.",
            "language": "en",
        }
    )
    assert example.label_source == "rules"
    assert example.rule_score > 0
    assert example.label_confidence == 0.0


def test_rule_labeling_is_not_influenced_by_persisted_feedback() -> None:
    labeler = DualWeakLabeler()
    assert labeler.priority_tagger.feedback_data["tag_corrections"] == {}
    assert labeler.priority_tagger.feedback_data["sender_preferences"] == {}


def test_route_populations_uses_rule_score_not_confidence() -> None:
    labeler = DualWeakLabeler(high_threshold=10.0, low_threshold=5.0)
    examples = [
        labeler.evaluate_email(
            {"id": "enron_high", "source": "enron", "subject": "URGENT", "body": "Please act immediately today.", "language": "en"}
        ),
        labeler.evaluate_email(
            {"id": "enron_low", "source": "enron", "subject": "Hello", "body": "Just sharing a note.", "language": "en"}
        ),
    ]
    high, ambiguous, low = labeler.route_populations(examples)
    assert len(high) + len(ambiguous) + len(low) == 2
    assert all(example.label_confidence == 0.0 for example in examples)
    assert all(example.rule_score >= 10.0 for example in high)
    assert all(example.rule_score < 5.0 for example in low)


def test_process_dataset_rejects_empty_records_without_aborting(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "weak.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "enron_1", "source": "enron", "subject": "Hello", "body": "A real message", "language": "en"}),
                json.dumps({"id": "enron_2", "source": "enron", "subject": "", "body": "", "language": "en"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = process_dataset_file(source, output)
    assert summary["total_records"] == 2
    assert summary["labeled_records"] == 1
    assert summary["rejected_records"] == 1
    assert output.is_file()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
