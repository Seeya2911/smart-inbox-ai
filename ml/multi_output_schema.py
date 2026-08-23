"""Canonical schema for the production Smart Inbox intent + priority dataset.

The legacy ``ml.schema`` module remains intentionally unchanged for the historical
information/request benchmark. This module is the new production data contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

INTENTS = {
    "request", "question", "meeting", "notification", "promotion",
    "complaint", "follow_up", "information", "other",
}
PRIORITIES = {"low", "medium", "high"}
LABEL_SOURCES = {"human", "llm", "rules", "user_feedback", "mixed"}
SOURCES = {"enron", "spam_corpus", "synthetic", "user_inbox", "other"}


@dataclass(frozen=True)
class MultiOutputEmailExample:
    """One provenance-preserving supervised or pseudo-labeled email example."""

    id: str
    subject: str
    body: str
    intent: str
    priority: str
    priority_reasons: List[str] = field(default_factory=list)
    source: str = "other"
    source_example_id: str = ""
    source_split: str = "unspecified"
    label_source: str = "human"
    label_confidence: float = 0.0
    rule_score: float | None = None
    is_synthetic: bool = False
    language: str = "en"
    provenance: str = ""

    def __post_init__(self) -> None:
        for name, value in (("id", self.id), ("subject", self.subject), ("body", self.body)):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        if not self.id.strip():
            raise ValueError("id must not be empty")
        if not (self.subject.strip() or self.body.strip()):
            raise ValueError("subject or body must contain text")
        if self.intent not in INTENTS:
            raise ValueError(f"Unsupported intent: {self.intent!r}")
        if self.priority not in PRIORITIES:
            raise ValueError(f"Unsupported priority: {self.priority!r}")
        if self.source not in SOURCES:
            raise ValueError(f"Unsupported source: {self.source!r}")
        if self.label_source not in LABEL_SOURCES:
            raise ValueError(f"Unsupported label_source: {self.label_source!r}")
        if not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError("label_confidence must be between 0 and 1")
        if self.rule_score is not None and self.rule_score < 0:
            raise ValueError("rule_score must be non-negative when present")
        if self.source == "synthetic" and not self.is_synthetic:
            raise ValueError("source='synthetic' requires is_synthetic=true")
        if self.is_synthetic and self.source != "synthetic":
            raise ValueError("is_synthetic=true requires source='synthetic'")

    @property
    def text(self) -> str:
        subject = self.subject.strip()
        body = self.body.strip()
        if subject and body:
            return f"Subject: {subject}\nBody: {body}"
        return subject or body

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiOutputEmailExample":
        required = {"id", "subject", "body", "intent", "priority", "source", "label_source"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return cls(
            id=str(data["id"]).strip(),
            subject=str(data.get("subject", "")).strip(),
            body=str(data.get("body", "")).strip(),
            intent=str(data["intent"]).lower().strip(),
            priority=str(data["priority"]).lower().strip(),
            priority_reasons=[str(x).strip() for x in data.get("priority_reasons", []) if str(x).strip()],
            source=str(data.get("source", "other")).lower().strip(),
            source_example_id=str(data.get("source_example_id", "")).strip(),
            source_split=str(data.get("source_split", "unspecified")).strip(),
            label_source=str(data["label_source"]).lower().strip(),
            label_confidence=float(data.get("label_confidence", 0.0)),
            rule_score=None if data.get("rule_score") is None else float(data["rule_score"]),
            is_synthetic=bool(data.get("is_synthetic", False)),
            language=str(data.get("language", "en")).lower().strip(),
            provenance=str(data.get("provenance", "")).strip(),
        )
