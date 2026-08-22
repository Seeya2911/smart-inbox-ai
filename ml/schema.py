"""Canonical training representation schema for the Smart Inbox INTENT task.

This module defines the canonical dataset schema for intent model training.
Per project requirements, urgency and priority labels are intentionally NOT
part of this stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

ALLOWED_INTENTS = {
    "request",
    "question",
    "meeting",
    "notification",
    "promotion",
    "complaint",
    "follow_up",
    "information",
    "other",
}

SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "unknown"}


@dataclass(frozen=True)
class CanonicalIntentExample:
    """Canonical representation of a single intent training example."""

    text: str
    language: str
    canonical_intent: str
    source_dataset: str
    source_example_id: str
    original_label: str
    original_split: str = "unspecified"
    source_group_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Canonical text must be a non-empty string")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language!r}")
        if self.canonical_intent not in ALLOWED_INTENTS:
            raise ValueError(
                f"Unsupported canonical_intent: {self.canonical_intent!r}. Must be one of {sorted(ALLOWED_INTENTS)}"
            )
        if not isinstance(self.source_dataset, str) or not self.source_dataset.strip():
            raise ValueError("source_dataset must be a non-empty string")
        if not isinstance(self.source_example_id, str) or not self.source_example_id.strip():
            raise ValueError("source_example_id must be a non-empty string")
        if not isinstance(self.original_label, str):
            raise ValueError("original_label must be a string")

    def to_dict(self) -> Dict[str, Any]:
        """Convert canonical example to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalIntentExample:
        """Validate and create a CanonicalIntentExample from a dictionary.

        Rejects records attempting to inject urgency or priority into the canonical intent schema.
        """
        forbidden = {"urgency", "priority"} & set(data.keys())
        if forbidden:
            raise ValueError(
                f"Forbidden fields present in canonical intent schema: {sorted(forbidden)}. "
                "Urgency and priority are intentionally excluded from the intent training stage."
            )

        text = str(data.get("text", "")).strip()
        if not text and ("subject" in data or "body" in data):
            subject = str(data.get("subject", "")).strip()
            body = str(data.get("body", "")).strip()
            text = f"Subject: {subject}\nBody: {body}".strip() if subject else body

        return cls(
            text=text,
            language=str(data.get("language", "unknown")).lower().strip(),
            canonical_intent=str(data.get("canonical_intent", data.get("intent", ""))).lower().strip(),
            source_dataset=str(data.get("source_dataset", "")).strip(),
            source_example_id=str(data.get("source_example_id", data.get("id", data.get("source_id", "")))).strip(),
            original_label=str(data.get("original_label", data.get("action_intent", data.get("source_intent", "")))),
            original_split=str(data.get("original_split", data.get("source_split", "unspecified"))).strip(),
            source_group_id=str(data.get("source_group_id", data.get("source_file", ""))).strip(),
        )
