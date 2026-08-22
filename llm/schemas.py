"""Typed, provider-independent output schema for email analysis."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "unknown"}
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
ALLOWED_URGENCY = {"low", "medium", "high", "critical"}
ALLOWED_SENTIMENT = {"negative", "neutral", "positive"}
ALLOWED_PRIORITY = {"low", "medium", "high", "critical"}


@dataclass(frozen=True)
class EmailAnalysis:
    """Normalized result returned by every LLM provider."""

    summary: str
    intent: str
    urgency: str
    sentiment: str
    priority: str
    language: str = "unknown"
    language_confidence: float = 0.0
    action_items: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    model: str = "unknown"

    def __post_init__(self) -> None:
        if self.intent not in ALLOWED_INTENTS:
            raise ValueError(f"Unsupported intent: {self.intent}")
        if self.urgency not in ALLOWED_URGENCY:
            raise ValueError(f"Unsupported urgency: {self.urgency}")
        if self.sentiment not in ALLOWED_SENTIMENT:
            raise ValueError(f"Unsupported sentiment: {self.sentiment}")
        if self.priority not in ALLOWED_PRIORITY:
            raise ValueError(f"Unsupported priority: {self.priority}")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= self.language_confidence <= 1.0:
            raise ValueError("language_confidence must be between 0 and 1")
        if not self.summary.strip():
            raise ValueError("summary must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model: str = "unknown") -> "EmailAnalysis":
        """Validate and normalize provider JSON into the application schema."""
        return cls(
            summary=str(data.get("summary", "")).strip(),
            intent=str(data.get("intent", "other")).lower().strip(),
            urgency=str(data.get("urgency", "low")).lower().strip(),
            sentiment=str(data.get("sentiment", "neutral")).lower().strip(),
            priority=str(data.get("priority", "medium")).lower().strip(),
            language=str(data.get("language", "unknown")).lower().strip(),
            language_confidence=float(data.get("language_confidence", 0.0)),
            action_items=[str(x).strip() for x in data.get("action_items", []) if str(x).strip()],
            entities=[str(x).strip() for x in data.get("entities", []) if str(x).strip()],
            reasoning=str(data.get("reasoning", "")).strip(),
            confidence=float(data.get("confidence", 0.0)),
            model=str(data.get("model", model)),
        )
