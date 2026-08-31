"""Data schemas for Smart Inbox AI local generation and task extraction."""
from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

ALLOWED_ACTION_TYPES: Set[str] = {
    "none",
    "reply",
    "create_task",
    "create_reminder",
    "create_calendar_event",
    "review_document",
    "contact_sender",
    "follow_up",
}


@dataclass
class SuggestedAction:
    """Structured action extracted from an email message.

    Safety: Actions are suggested only and require explicit user confirmation
    before any external operation or calendar execution occurs.
    """

    action_type: str = "none"
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[str] = None
    due_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    participants: List[str] = field(default_factory=list)
    source_evidence: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action_type not in ALLOWED_ACTION_TYPES:
            # Fallback to none for unsupported types
            self.action_type = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuggestedAction":
        action_type = str(data.get("action_type", "none")).lower().strip()
        if action_type not in ALLOWED_ACTION_TYPES:
            action_type = "none"

        duration = data.get("duration_minutes")
        if duration is not None:
            try:
                duration = int(duration)
            except (ValueError, TypeError):
                duration = None

        return cls(
            action_type=action_type,
            title=data.get("title"),
            description=data.get("description"),
            due_date=data.get("due_date"),
            due_time=data.get("due_time"),
            duration_minutes=duration,
            participants=list(data.get("participants", []) or []),
            source_evidence=data.get("source_evidence"),
        )


@dataclass
class GenerationOutput:
    """Consolidated generation output for a single email, preserving raw and parsed outputs."""

    summary: str
    action: SuggestedAction
    raw_model_summary: str = ""
    raw_model_action: str = ""
    intent_context: Optional[str] = None
    priority_context: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "action": self.action.to_dict(),
            "raw_model_summary": self.raw_model_summary,
            "raw_model_action": self.raw_model_action,
            "intent_context": self.intent_context,
            "priority_context": self.priority_context,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class GenerationTrainingExample:
    """Schema for future supervised generation fine-tuning datasets."""

    id: str
    subject: str
    body: str
    intent: str
    priority: str
    summary_target: str
    action_target: SuggestedAction
    label_source: Literal["teacher", "human", "user_feedback"] = "human"
    source: str = "canonical"
    is_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "body": self.body,
            "intent": self.intent,
            "priority": self.priority,
            "summary_target": self.summary_target,
            "action_target": self.action_target.to_dict(),
            "label_source": self.label_source,
            "source": self.source,
            "is_synthetic": self.is_synthetic,
        }


@dataclass
class UserFeedbackExample:
    """Schema for recording user preference/feedback on summaries and actions."""

    email_id: str
    generated_summary: str
    generated_action: Dict[str, Any]
    user_accepted_summary: bool
    user_accepted_action: bool
    user_edited_summary: Optional[str] = None
    user_edited_action: Optional[Dict[str, Any]] = None
    feedback_notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
