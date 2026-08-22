"""LLM provider abstractions.

The application depends on this small interface instead of a vendor-specific SDK.
OpenAI-compatible providers can be used for hosted or local inference.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from .schemas import EmailAnalysis


SYSTEM_PROMPT = """You are a multilingual email intelligence system.

Read the ENTIRE supplied email (subject and body) and infer meaning from context.
Do NOT classify using keyword presence alone. In particular, do not mark an email
urgent merely because words such as 'urgent', 'asap', or their translations occur.
Consider deadlines, requested actions, consequences, explicit and implicit intent,
negation, modality, politeness, discourse context, and the relationship between
sentences. Preserve the meaning of negation (for example, 'not urgent').

First identify the language of the email. Supported languages are English (en),
German (de), French (fr), and Spanish (es). If the message is too short or genuinely
ambiguous, use 'unknown' rather than inventing a language.

Return ONLY valid JSON with these keys:
summary, intent, urgency, sentiment, priority, language, language_confidence,
action_items, entities, reasoning, confidence.

intent must be one of: request, question, meeting, notification, promotion, complaint,
follow_up, information, other.
urgency and priority must be one of: low, medium, high, critical.
sentiment must be one of: negative, neutral, positive.
language must be one of: en, de, fr, es, unknown.
confidence and language_confidence must be numbers from 0 to 1.
Do not invent facts, deadlines, entities, or actions that are not supported by the email.
Keep the summary concise. Keep action_items empty when there are no concrete actions.
Reasoning should briefly state the contextual evidence used for classification, not hidden
chain-of-thought or private reasoning.
"""


class LLMProvider(ABC):
    """Provider contract used by the email analysis layer."""

    @abstractmethod
    def analyze(self, message: Dict[str, Any]) -> EmailAnalysis:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """Call any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI-compatible provider requires the 'openai' package."
            ) from exc

        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=api_key or os.getenv("LLM_API_KEY"),
            base_url=base_url or os.getenv("LLM_BASE_URL"),
        )

    def analyze(self, message: Dict[str, Any]) -> EmailAnalysis:
        subject = str(message.get("subject", "")).strip()
        body = str(message.get("message_text", message.get("body", ""))).strip()
        if not body and not subject:
            raise ValueError("Cannot analyze an empty message")

        user_content = f"Subject: {subject}\n\nBody:\n{body}".strip()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid JSON") from exc
        return EmailAnalysis.from_dict(payload, model=self.model)


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests and credential-free demos.

    This provider intentionally uses simple rules and is explicitly NOT an LLM.
    It exists to test the application contract and provide a reproducible baseline.
    """

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model

    def analyze(self, message: Dict[str, Any]) -> EmailAnalysis:
        subject = str(message.get("subject", "")).strip()
        body = str(message.get("message_text", message.get("body", ""))).strip()
        text = f"{subject} {body}".strip()
        lowered = text.lower()

        if any(word in lowered for word in ("urgent", "asap", "immediately")):
            urgency = priority = "high"
        else:
            urgency = priority = "medium"

        if "meeting" in lowered or "calendar" in lowered:
            intent = "meeting"
        elif "please" in lowered or "can you" in lowered:
            intent = "request"
        elif "?" in text:
            intent = "question"
        else:
            intent = "information"

        sentiment = "negative" if any(w in lowered for w in ("problem", "failed", "angry")) else "neutral"
        summary = body[:180] + ("..." if len(body) > 180 else "")
        return EmailAnalysis(
            summary=summary or subject or "No message content",
            intent=intent,
            urgency=urgency,
            sentiment=sentiment,
            priority=priority,
            language="unknown",
            language_confidence=0.0,
            reasoning="Deterministic mock provider used for testing; not an LLM prediction.",
            confidence=0.5,
            model=self.model,
        )
