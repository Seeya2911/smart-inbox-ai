"""Unit tests for local generation foundation, summarization, action extraction, and schemas."""
import json
from unittest.mock import MagicMock

import pytest

from ml.generation.inference import extract_action, parse_action_output, process_email, summarize_email
from ml.generation.prompts import build_action_extraction_prompt, build_summarization_prompt, clean_email_text
from ml.generation.schemas import (
    ALLOWED_ACTION_TYPES,
    GenerationOutput,
    GenerationTrainingExample,
    SuggestedAction,
    UserFeedbackExample,
)


class TestGenerationSchemas:
    def test_suggested_action_defaults_and_validation(self) -> None:
        action = SuggestedAction(action_type="create_task", title="Submit expense report")
        assert action.action_type == "create_task"
        assert action.title == "Submit expense report"
        assert action.due_date is None
        assert action.participants == []

        # Invalid action type falls back to none
        invalid = SuggestedAction(action_type="invalid_action_type")
        assert invalid.action_type == "none"

    def test_suggested_action_from_dict(self) -> None:
        data = {
            "action_type": "create_calendar_event",
            "title": "Strategy Sync",
            "due_date": "tomorrow",
            "due_time": "3:00 PM",
            "duration_minutes": "45",
            "participants": ["alex@example.com"],
        }
        action = SuggestedAction.from_dict(data)
        assert action.action_type == "create_calendar_event"
        assert action.duration_minutes == 45
        assert len(action.participants) == 1

    def test_generation_training_example_schema(self) -> None:
        action = SuggestedAction(action_type="reply", title="Reply with confirmation")
        ex = GenerationTrainingExample(
            id="synth_001",
            subject="Invoice received",
            body="Please confirm receipt.",
            intent="transactional",
            priority="low",
            summary_target="Invoice was received and confirmation requested.",
            action_target=action,
        )
        d = ex.to_dict()
        assert d["id"] == "synth_001"
        assert d["action_target"]["action_type"] == "reply"

    def test_user_feedback_example_schema(self) -> None:
        fb = UserFeedbackExample(
            email_id="inbox_102",
            generated_summary="Team meeting at 2pm.",
            generated_action={"action_type": "create_calendar_event"},
            user_accepted_summary=True,
            user_accepted_action=False,
            user_edited_action={"action_type": "none"},
        )
        d = fb.to_dict()
        assert d["email_id"] == "inbox_102"
        assert d["user_accepted_summary"] is True
        assert d["user_accepted_action"] is False


class TestPrompts:
    def test_build_summarization_prompt_with_context(self) -> None:
        prompt = build_summarization_prompt(
            subject="Server Alert",
            body="CPU usage exceeded 95%.",
            intent="security",
            priority="high",
        )
        assert "Category: security" in prompt
        assert "Priority: high" in prompt
        assert "Server Alert" in prompt

    def test_build_action_prompt_cleans_boilerplate(self) -> None:
        body = "Please review the attached contract.\n--\nBest regards,\nJohn Doe"
        prompt = build_action_extraction_prompt("Contract Review", body)
        assert "Contract Review" in prompt
        assert "Best regards" not in prompt


class TestActionParsing:
    def test_parse_calendar_event_with_date_and_time(self) -> None:
        raw = "create_calendar_event: sync with team tomorrow at 3pm"
        action = parse_action_output(raw, "Project sync", "Let's meet tomorrow at 3pm.")
        assert action.action_type == "create_calendar_event"
        assert action.due_date is not None
        assert "tomorrow" in action.due_date.lower()
        assert action.due_time is not None
        assert "3pm" in action.due_time.lower()

    def test_parse_non_actionable_notification(self) -> None:
        raw = "none: purely informational system broadcast"
        action = parse_action_output(raw, "System Update", "System update completed successfully.")
        assert action.action_type == "none"

    def test_parse_reply_action(self) -> None:
        raw = "reply: answer questions regarding the proposal"
        action = parse_action_output(raw, "Proposal Query", "Could you clarify line 4?")
        assert action.action_type == "reply"

    def test_parse_empty_or_malformed_output(self) -> None:
        action = parse_action_output("", "Hello", "Just checking in.")
        assert action.action_type in ALLOWED_ACTION_TYPES


class TestInferencePipeline:
    def test_ultra_short_email_handling(self) -> None:
        mock_model = MagicMock()
        summary, raw_summary = summarize_email("OK", "Thanks!", model=mock_model)
        assert "OK" in summary
        assert "OK" in raw_summary
        assert len(summary) > 0

    def test_process_email_end_to_end_with_mock_model(self) -> None:
        mock_model = MagicMock()
        mock_model.generate.side_effect = [
            "Password reset request required for security verification.",
            "create_task: reset password before Friday EOD",
        ]
        out = process_email(
            subject="Urgent Security Alert",
            body="Please reset your password before Friday at 5pm.",
            intent="security",
            priority="high",
            model=mock_model,
        )
        assert isinstance(out, GenerationOutput)
        assert "password" in out.summary.lower()
        assert out.raw_model_summary == "Password reset request required for security verification."
        assert out.raw_model_action == "create_task: reset password before Friday EOD"
        assert out.action.action_type in ["create_task", "create_reminder", "reply"]
        assert out.action.due_date is not None
        assert "friday" in out.action.due_date.lower()
