"""Regression tests for the legacy baseline kept beside the LLM stack."""

from datetime import datetime, timezone

from smart_summarizer_v3 import SmartSummarizerV3, summarize_message


def _message(text: str) -> dict:
    return {
        "user_id": "regression-user",
        "platform": "email",
        "message_text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def test_request_question_is_not_misclassified():
    result = SmartSummarizerV3().summarize(_message("Can you send me the report?"), use_context=False)
    assert result["intent"] == "request"


def test_convenience_wrapper_accepts_context_flag():
    result = summarize_message(_message("Please review the report."), use_context=False)
    assert result["intent"] == "request"


def test_context_survives_reloading():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        context_file = str(Path(directory) / "context.json")
        first = SmartSummarizerV3(context_file=context_file)
        first.summarize(_message("Please review this document."), use_context=True)

        second = SmartSummarizerV3(context_file=context_file)
        assert second.get_user_context("regression-user", "email")


def test_time_bounded_deadline_is_high_urgency():
    result = SmartSummarizerV3().summarize(
        _message("Any update? The presentation is in 2 hours!"), use_context=False
    )
    assert result["urgency"] == "high"
