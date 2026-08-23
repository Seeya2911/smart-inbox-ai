import pytest

from ml.multi_output_schema import INTENTS, PRIORITIES, MultiOutputEmailExample


def valid(**overrides):
    data = {
        "id": "e1",
        "subject": "Password changed",
        "body": "If this was not you, secure your account now.",
        "intent": "notification",
        "priority": "high",
        "source": "synthetic",
        "label_source": "llm",
        "label_confidence": 0.92,
        "is_synthetic": True,
    }
    data.update(overrides)
    return MultiOutputEmailExample(**data)


def test_schema_exposes_target_taxonomy():
    assert len(INTENTS) == 9
    assert INTENTS == {
        "request", "question", "meeting", "notification", "promotion",
        "complaint", "follow_up", "information", "other",
    }
    assert PRIORITIES == {"low", "medium", "high"}


def test_subject_and_body_are_combined_for_model_text():
    example = valid()
    assert example.text == "Subject: Password changed\nBody: If this was not you, secure your account now."


def test_round_trip_preserves_provenance():
    example = valid(priority_reasons=["security", "action_required"], rule_score=4.5)
    restored = MultiOutputEmailExample.from_dict(example.to_dict())
    assert restored == example


def test_rejects_invalid_labels():
    with pytest.raises(ValueError, match="Unsupported intent"):
        valid(intent="security")
    with pytest.raises(ValueError, match="Unsupported priority"):
        valid(priority="critical")


def test_rejects_synthetic_provenance_mismatch():
    with pytest.raises(ValueError, match="requires is_synthetic=true"):
        valid(is_synthetic=False)
    with pytest.raises(ValueError, match="requires source='synthetic'"):
        valid(source="enron", is_synthetic=True)


def test_rule_score_is_not_treated_as_confidence():
    example = valid(label_source="rules", label_confidence=0.0, rule_score=8.0)
    assert example.rule_score == 8.0
    assert example.label_confidence == 0.0
