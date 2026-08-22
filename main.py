"""Smart Inbox AI application entry point.

The modern application uses an LLM provider for semantic email analysis and
keeps deterministic components as explicit baselines rather than pretending
that keyword rules are an LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from llm import EmailAnalyzer, MockLLMProvider, OpenAICompatibleProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SmartInboxAssistant:
    """Application service for LLM-powered inbox analysis."""

    def __init__(self, analyzer: EmailAnalyzer | None = None) -> None:
        self.analyzer = analyzer or self._build_default_analyzer()

    @staticmethod
    def _build_default_analyzer() -> EmailAnalyzer:
        # A credential-free mock is the safe default for demos and CI.
        if not os.getenv("LLM_API_KEY"):
            return EmailAnalyzer(MockLLMProvider())
        return EmailAnalyzer(OpenAICompatibleProvider())

    def process_message(self, message_data: Dict[str, Any], platform: str = "email") -> Dict[str, Any]:
        payload = dict(message_data)
        payload["platform"] = platform
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now().isoformat()
        analysis = self.analyzer.analyze(payload)
        result = analysis.to_dict()
        result.update({
            "message_id": payload.get("message_id"),
            "platform": platform,
            "processed_at": datetime.now().isoformat(),
        })
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Inbox AI")
    parser.add_argument("--demo", action="store_true", help="Run a credential-free synthetic example")
    parser.add_argument("--subject", default="", help="Email subject")
    parser.add_argument("--message", default="", help="Email body")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock analysis")
    args = parser.parse_args()

    analyzer = EmailAnalyzer(MockLLMProvider()) if args.mock or args.demo else None
    assistant = SmartInboxAssistant(analyzer)

    if args.demo:
        subject = "Urgent: project meeting tomorrow"
        body = "Please join the project meeting tomorrow at 10:00. We need your review ASAP."
    else:
        subject = args.subject or input("Subject: ").strip()
        body = args.message or input("Message: ").strip()

    result = assistant.process_message({
        "subject": subject,
        "message_text": body,
        "platform": "email",
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
