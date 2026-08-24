"""Canonical training representation schema for Smart Inbox AI multi-output NLP pipeline.

Defines the canonical dataset schema for multi-output training (INTENT + PRIORITY),
supporting full provenance tracking, ID namespacing, and weak labeling metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Optional

ALLOWED_INTENTS: Set[str] = {
    "request", "question", "meeting", "notification", "promotion",
    "complaint", "follow_up", "information", "security", "transactional", "other",
}

ALLOWED_PRIORITIES: Set[str] = {"high", "medium", "low"}

ALLOWED_LABEL_SOURCES: Set[str] = {"rules", "llm", "human", "user_feedback", "ground_truth"}
VALID_SOURCE_PREFIXES: Set[str] = {"enron", "spam", "phishing", "synthetic", "inbox"}
SUPPORTED_LANGUAGES: Set[str] = {"en", "de", "fr", "es", "unknown"}


def format_namespaced_id(source: str, raw_id: str) -> str:
    """Format raw example ID with a source namespace prefix."""
    raw = str(raw_id).strip()
    if not raw:
        raw = "000000"
    for prefix in VALID_SOURCE_PREFIXES:
        if raw.startswith(f"{prefix}_"):
            return raw

    src_lower = str(source).lower().strip()
    prefix = "inbox"
    if "enron" in src_lower:
        prefix = "enron"
    elif "spam" in src_lower:
        prefix = "spam"
    elif "phishing" in src_lower:
        prefix = "phishing"
    elif "synthetic" in src_lower:
        prefix = "synthetic"
    return f"{prefix}_{raw}"


@dataclass(frozen=True)
class CanonicalEmailExample:
    """Unified canonical multi-output example representation."""

    id: str
    subject: str
    body: str
    intent: str
    priority: str
    priority_reasons: List[str] = field(default_factory=list)
    source: str = "synthetic"
    label_source: str = "rules"
    label_confidence: float = 1.0
    rule_score: float = 0.0
    language: str = "en"
    source_group_id: str = ""
    is_synthetic: bool = False
    provenance: str = ""
    # Independent rule signal. These fields are metadata, never training features.
    rule_intent: Optional[str] = None
    rule_priority: Optional[str] = None
    llm_rule_agreement: Optional[bool] = None
    llm_intent_reason: str = ""
    llm_priority_reason: str = ""
    label_resolution_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Example id must be a non-empty string")
        if not any(self.id.startswith(f"{p}_") for p in VALID_SOURCE_PREFIXES):
            raise ValueError(f"Example id '{self.id}' must be namespaced with one of {sorted(VALID_SOURCE_PREFIXES)}")
        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError("Email body must be a non-empty string")
        if self.intent not in ALLOWED_INTENTS:
            raise ValueError(f"Unsupported intent: {self.intent!r}. Must be one of {sorted(ALLOWED_INTENTS)}")
        if self.priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"Unsupported priority: {self.priority!r}. Must be one of {sorted(ALLOWED_PRIORITIES)}")
        if self.rule_intent is not None and self.rule_intent not in ALLOWED_INTENTS:
            raise ValueError(f"Unsupported rule_intent: {self.rule_intent!r}")
        if self.rule_priority is not None and self.rule_priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"Unsupported rule_priority: {self.rule_priority!r}")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language!r}")
        if not (0.0 <= self.label_confidence <= 1.0):
            raise ValueError(f"label_confidence must be between 0.0 and 1.0; got {self.label_confidence}")

    @property
    def full_text(self) -> str:
        """Full email text combined from subject and body."""
        sbj = self.subject.strip()
        bdy = self.body.strip()
        return f"Subject: {sbj}\nBody: {bdy}".strip() if sbj else bdy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalEmailExample":
        source = str(data.get("source", data.get("source_dataset", "synthetic"))).strip()
        raw_id = str(data.get("id", data.get("source_example_id", "000000"))).strip()
        namespaced_id = format_namespaced_id(source, raw_id)

        subject = str(data.get("subject", "")).strip()
        body = str(data.get("body", data.get("text", ""))).strip()
        if not body and "text" in data:
            body = str(data["text"]).strip()

        intent = str(data.get("intent", data.get("canonical_intent", "other"))).lower().strip()
        priority = str(data.get("priority", "low")).lower().strip()

        reasons = data.get("priority_reasons", [])
        if isinstance(reasons, str):
            reasons = [reasons]
        priority_reasons = [str(r).strip() for r in reasons if str(r).strip()]

        label_source = str(data.get("label_source", "rules")).lower().strip()
        if label_source not in ALLOWED_LABEL_SOURCES:
            label_source = "rules"

        try:
            confidence = float(data.get("label_confidence", 1.0))
        except (ValueError, TypeError):
            confidence = 1.0
        try:
            rule_score = float(data.get("rule_score", 0.0))
        except (ValueError, TypeError):
            rule_score = 0.0

        return cls(
            id=namespaced_id,
            subject=subject,
            body=body,
            intent=intent,
            priority=priority,
            priority_reasons=priority_reasons,
            source=source,
            label_source=label_source,
            label_confidence=max(0.0, min(1.0, confidence)),
            rule_score=rule_score,
            language=str(data.get("language", "en")).lower().strip(),
            source_group_id=str(data.get("source_group_id", data.get("group_id", ""))).strip(),
            is_synthetic=bool(data.get("is_synthetic", "synthetic" in source.lower())),
            provenance=str(data.get("provenance", source)).strip(),
            rule_intent=(str(data["rule_intent"]).lower().strip() if data.get("rule_intent") is not None else None),
            rule_priority=(str(data["rule_priority"]).lower().strip() if data.get("rule_priority") is not None else None),
            llm_rule_agreement=(bool(data["llm_rule_agreement"]) if data.get("llm_rule_agreement") is not None else None),
            llm_intent_reason=str(data.get("llm_intent_reason", "")).strip(),
            llm_priority_reason=str(data.get("llm_priority_reason", "")).strip(),
            label_resolution_reason=str(data.get("label_resolution_reason", "")).strip(),
        )


@dataclass(frozen=True)
class CanonicalIntentExample:
    """Legacy backward-compatible wrapper for single-intent example schema."""

    text: str
    language: str
    canonical_intent: str
    source_dataset: str
    source_example_id: str
    original_label: str
    original_split: str = "unspecified"
    source_group_id: str = ""
    is_synthetic: bool = False
    provenance: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Canonical text must be a non-empty string")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language!r}")
        if self.canonical_intent not in ALLOWED_INTENTS:
            raise ValueError(f"Unsupported canonical_intent: {self.canonical_intent!r}. Must be one of {sorted(ALLOWED_INTENTS)}")
        if not isinstance(self.source_dataset, str) or not self.source_dataset.strip():
            raise ValueError("source_dataset must be a non-empty string")
        if not isinstance(self.source_example_id, str) or not self.source_example_id.strip():
            raise ValueError("source_example_id must be a non-empty string")
        if not isinstance(self.original_label, str):
            raise ValueError("original_label must be a string")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalIntentExample":
        forbidden = {"urgency", "priority"} & set(data.keys())
        if forbidden:
            raise ValueError(f"Forbidden fields present in canonical intent schema: {sorted(forbidden)}. Urgency and priority are intentionally excluded from the intent training stage.")

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
            is_synthetic=bool(data.get("is_synthetic", False)),
            provenance=str(data.get("provenance", "")).strip(),
        )
