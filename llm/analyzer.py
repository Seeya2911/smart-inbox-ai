"""Application-facing LLM email analyzer."""

from __future__ import annotations

from typing import Any, Dict

from .language import detect_language
from .provider import LLMProvider
from .schemas import EmailAnalysis


class EmailAnalyzer:
    """Validate input, detect language, then delegate semantic analysis to an LLM."""

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

        # Detect language independently before the LLM sees the message. The original
        # text is preserved; no stop-word removal, stemming, or lemmatization is applied.
        text = f"{normalized['subject']} {normalized['message_text']}".strip()
        detected = detect_language(text)
        enriched = {
            **normalized,
            "detected_language": detected.language,
            "detected_language_confidence": detected.confidence,
        }
        result = self.provider.analyze(enriched)

        disagreement = (
            detected.supported
            and result.language in {"en", "de", "fr", "es"}
            and result.language != detected.language
        )
        return EmailAnalysis(
            **{
                **result.to_dict(),
                "detected_language": detected.language,
                "detected_language_confidence": detected.confidence,
                "language_disagreement": disagreement,
            }
        )
