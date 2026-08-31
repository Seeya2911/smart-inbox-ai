import pytest

from ml.llm_labeler import label_example
from ml.schema import CanonicalEmailExample


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def classify(self, subject, body):
        return self.payload


def make_example():
    return CanonicalEmailExample(
        id="enron_001",
        subject="Team lunch next week",
        body="Let me know which day works best for lunch.",
        intent="information",
        priority="low",
        source="enron",
        source_example_id="001",
        source_split="train",
        source_group_id="thread-1",
        label_source="rules",
        provenance="corbt/enron-emails",
    )


def test_llm_is_authoritative_when_rule_disagrees():
    example = make_example()
    llm = FakeLLM(
        {
            "intent": "security",
            "priority": "high",
            "priority_reasons": ["potential account compromise"],
            "intent_reason": "The message reports suspicious account activity.",
            "priority_reason": "Account compromise requires prompt action.",
            "confidence": 0.91,
        }
    )

    labeled = label_example(example, llm)

    assert labeled.intent == "security"
    assert labeled.priority == "high"
    assert labeled.label_source == "llm"
    assert labeled.rule_intent != "security"
    assert labeled.llm_rule_agreement is False
    assert labeled.label_confidence == pytest.approx(0.91)
    assert labeled.label_resolution_reason
    assert "LLM intent 'security' was selected over rule intent" in labeled.label_resolution_reason


def test_llm_label_wins_and_disagreement_is_recorded():
    example = CanonicalEmailExample(
        id="enron_002",
        subject="Please review the account details",
        body="Could you review the account details and let me know if they are correct?",
        intent="request",
        priority="low",
        source="enron",
        source_example_id="002",
        source_split="train",
        label_source="rules",
    )
    llm = FakeLLM(
        {
            "intent": "question",
            "priority": "medium",
            "priority_reasons": ["a response is requested"],
            "intent_reason": "The sender primarily asks for confirmation.",
            "priority_reason": "A response is useful but not immediately critical.",
            "confidence": 0.84,
        }
    )

    labeled = label_example(example, llm)

    assert labeled.intent == "question"
    assert labeled.priority == "medium"
    assert labeled.rule_intent == "request"
    assert labeled.llm_rule_agreement is False
    assert "LLM intent 'question' was selected over rule intent 'request'" in labeled.label_resolution_reason
    assert "primarily asks for confirmation" in labeled.label_resolution_reason
    assert labeled.source_example_id == "002"
    assert labeled.source_split == "train"
    assert labeled.provenance == example.provenance


def test_invalid_llm_intent_is_rejected():
    with pytest.raises(ValueError, match="unsupported intent"):
        label_example(
            make_example(),
            FakeLLM(
                {
                    "intent": "spammy",
                    "priority": "low",
                    "priority_reasons": [],
                    "intent_reason": "bad",
                    "priority_reason": "bad",
                    "confidence": 0.5,
                }
            ),
        )
