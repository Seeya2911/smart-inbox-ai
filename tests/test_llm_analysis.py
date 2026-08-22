import pytest

from llm import EmailAnalyzer, MockLLMProvider


def test_mock_provider_returns_structured_analysis():
    analyzer = EmailAnalyzer(MockLLMProvider())
    result = analyzer.analyze(
        {
            "subject": "Urgent meeting",
            "message_text": "Please join the meeting ASAP.",
            "platform": "email",
        }
    )

    assert result.summary
    assert result.intent == "meeting"
    assert result.urgency == "high"
    assert result.priority == "high"
    assert 0.0 <= result.confidence <= 1.0


def test_empty_message_is_rejected():
    analyzer = EmailAnalyzer(MockLLMProvider())
    with pytest.raises(ValueError):
        analyzer.analyze({"subject": "", "message_text": ""})


def test_invalid_provider_output_is_rejected():
    with pytest.raises(ValueError):
        from llm.schemas import EmailAnalysis

        EmailAnalysis(
            summary="test",
            intent="not-a-real-intent",
            urgency="low",
            sentiment="neutral",
            priority="low",
        )
