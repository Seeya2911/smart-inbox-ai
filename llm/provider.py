"""LLM provider abstractions.

The application depends on this small interface instead of a vendor-specific SDK.
OpenAI-compatible providers can be used for hosted or local inference (for example,
Ollama's OpenAI-compatible endpoint).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from .schemas import EmailAnalysis


SYSTEM_PROMPT = """You are an email analysis assistant.
Analyze the supplied email and return ONLY valid JSON with these keys:
summary, intent, urgency, sentiment, priority, action_items, entities, reasoning, confidence.

intent must be one of: request, question, meeting, notification, promotion, complaint,
follow_up, information, other.
urgency and priority must be one of: low, medium, high, critical.
sentiment must be one of: negative, neutral, positive.
confidence must be a number from 0 to 1.
Do not invent facts that are not present in the message.
Keep the summary concise and action_items empty when there are no concrete actions.
"""


class LLMProvider(ABC):
    """Provider contract used by the email analysis layer."""

    @abstractmethod
    def analyze(self, message: Dict[str, Any]) -> EmailAnalysis:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """Call any OpenAI-compatible chat-completions endpoint.

    Environment variables:
    - LLM_API_KEY
    - LLM_BASE_URL (optional; defaults to OpenAI)
    - LLM_MODEL (default: gpt-4o-mini)
    """

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
        payload = json.loads(content)
        return EmailAnalysis.from_dict(payload, model=self.model)


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests and credential-free demos."""

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
            reasoning="Deterministic mock provider used for testing; not an LLM prediction.",
            confidence=0.5,
            model=self.model,
        )
