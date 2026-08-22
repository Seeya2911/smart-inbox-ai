"""Application-facing LLM email analyzer."""

from __future__ import annotations

from typing import Any, Dict

from .provider import LLMProvider
from .schemas import EmailAnalysis


class EmailAnalyzer:
    """Normalize application messages and delegate analysis to an LLM provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def analyze(self, message: Dict[str, Any]) -> EmailAnalysis:
        normalized = {
            "subject": str(message.get("subject", "")).strip(),
            "message_text": str(message.get("message_text", message.get("body", ""))).strip(),
            "sender": str(message.get("sender", "")).strip(),
            "platform": str(message.get("platform", "email")).strip().lower(),
        }
        if not normalized["subject"] and not normalized["message_text"]:
            raise ValueError("Message must contain a subject or body")
        return self.provider.analyze(normalized)
