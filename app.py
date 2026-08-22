"""Modern Smart Inbox AI entry point.

Use ``--demo`` for a credential-free local run. Set LLM_API_KEY and optionally
LLM_BASE_URL/LLM_MODEL to enable an OpenAI-compatible LLM provider.
"""

from __future__ import annotations

import argparse
import json
import os

from llm import EmailAnalyzer, MockLLMProvider, OpenAICompatibleProvider


def build_analyzer(use_mock: bool = False) -> EmailAnalyzer:
    if use_mock or not os.getenv("LLM_API_KEY"):
        return EmailAnalyzer(MockLLMProvider())
    return EmailAnalyzer(OpenAICompatibleProvider())


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Inbox AI")
    parser.add_argument("--demo", action="store_true", help="Analyze a built-in synthetic email")
    parser.add_argument("--subject", default="", help="Email subject")
    parser.add_argument("--message", default="", help="Email body")
    parser.add_argument("--mock", action="store_true", help="Use the deterministic test provider")
    args = parser.parse_args()

    if args.demo:
        subject = "Urgent: project meeting tomorrow"
        message = "Please join the project meeting tomorrow at 10:00. We need your review ASAP."
    else:
        subject = args.subject or input("Subject: ").strip()
        message = args.message or input("Message: ").strip()

    analyzer = build_analyzer(use_mock=args.mock or args.demo)
    result = analyzer.analyze({"subject": subject, "message_text": message, "platform": "email"})
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
