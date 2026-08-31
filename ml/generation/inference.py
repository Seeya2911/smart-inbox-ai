"""Inference functions for email summarization and action extraction."""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, List, Optional

from ml.deduplication import strip_email_boilerplate
from ml.generation.model import LocalGenerationModel, load_model
from ml.generation.prompts import build_action_extraction_prompt, build_summarization_prompt
from ml.generation.schemas import ALLOWED_ACTION_TYPES, GenerationOutput, SuggestedAction

# Regex patterns to assist robust action attribute extraction
_DATE_EXTRACTION_RE = re.compile(
    r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
    re.IGNORECASE,
)

_TIME_EXTRACTION_RE = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|est|pst|cst|utc|gmt)|eod|end\s+of\s+day|noon|midnight)\b",
    re.IGNORECASE,
)


def parse_action_output(
    raw_model_text: str,
    subject: str,
    body: str,
    intent: Optional[str] = None,
) -> SuggestedAction:
    """Parse raw model text into a validated, structured SuggestedAction object.

    Uses deterministic fallback parsing to ensure strict adherence to ALLOWED_ACTION_TYPES
    and zero invented/hallucinated calendar attributes.
    """
    clean_raw = raw_model_text.strip().lower()
    full_text = f"{subject} {strip_email_boilerplate(body)}"

    # 1. Determine action_type
    action_type = "none"

    if any(k in clean_raw for k in ["none", "no action", "informational", "n/a", "no task"]):
        action_type = "none"
    elif "calendar" in clean_raw or "meeting" in clean_raw or "schedule" in clean_raw or intent == "meeting":
        action_type = "create_calendar_event"
    elif "reply" in clean_raw or "respond" in clean_raw or "answer" in clean_raw:
        action_type = "reply"
    elif "reminder" in clean_raw:
        action_type = "create_reminder"
    elif "review" in clean_raw or "read" in clean_raw:
        action_type = "review_document"
    elif "follow up" in clean_raw or "follow_up" in clean_raw or intent == "follow_up":
        action_type = "follow_up"
    elif "contact" in clean_raw or "call" in clean_raw:
        action_type = "contact_sender"
    elif any(k in clean_raw for k in ["task", "submit", "send", "action", "pay", "reset", "approve", "verify"]):
        action_type = "create_task"
    elif intent in ["request", "security"]:
        action_type = "create_task"

    if action_type == "none":
        return SuggestedAction(action_type="none")

    # 2. Extract Evidence & Title
    title = None
    if raw_model_text and len(raw_model_text) > 3 and not raw_model_text.lower().startswith("none"):
        title = raw_model_text[:120].strip()
    elif subject:
        title = f"Action for: {subject[:100].strip()}"

    # 3. Extract dates & times only if supported in text
    date_match = _DATE_EXTRACTION_RE.search(full_text)
    due_date = date_match.group(0).strip() if date_match else None

    time_match = _TIME_EXTRACTION_RE.search(full_text)
    due_time = time_match.group(0).strip() if time_match else None

    # 4. Extract evidence snippet
    evidence_snippet = None
    first_sentence = full_text.split(".")[0].strip()
    if len(first_sentence) > 10:
        evidence_snippet = first_sentence[:200]

    return SuggestedAction(
        action_type=action_type,
        title=title,
        description=f"Action required from email: {subject[:80]}",
        due_date=due_date,
        due_time=due_time,
        source_evidence=evidence_snippet,
    )


def summarize_email(
    subject: str,
    body: str,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    model: Optional[LocalGenerationModel] = None,
) -> str:
    """Generate a concise, factual summary of an email using the local FLAN-T5 model."""
    if model is None:
        model = load_model()

    clean_body = strip_email_boilerplate(body or "")
    if len(clean_body) < 15 and len(subject or "") < 15:
        # Handle ultra-short emails directly and gracefully
        return f"{subject.strip()}: {clean_body.strip()}".strip(" :")

    prompt = build_summarization_prompt(subject, body, intent=intent, priority=priority)
    summary = model.generate(
        prompt,
        max_new_tokens=96,
        min_length=8,
        num_beams=2,
        length_penalty=1.0,
    )

    # Post-clean summary output
    clean_summary = summary.strip()
    if not clean_summary:
        clean_summary = f"{subject.strip()}. {clean_body[:100].strip()}..."
    return clean_summary


def extract_action(
    subject: str,
    body: str,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    model: Optional[LocalGenerationModel] = None,
) -> SuggestedAction:
    """Extract a structured SuggestedAction from an email message using local FLAN-T5."""
    if model is None:
        model = load_model()

    clean_body = strip_email_boilerplate(body or "")
    # Non-actionable intents shortcut
    if intent in ["promotion", "other", "notification"] and not any(k in body.lower() for k in ["deadline", "action required", "urgent"]):
        return SuggestedAction(action_type="none")

    prompt = build_action_extraction_prompt(subject, body, intent=intent, priority=priority)
    raw_action_text = model.generate(
        prompt,
        max_new_tokens=64,
        min_length=2,
        num_beams=2,
    )

    structured_action = parse_action_output(raw_action_text, subject, body, intent=intent)
    return structured_action


def process_email(
    subject: str,
    body: str,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    model: Optional[LocalGenerationModel] = None,
) -> GenerationOutput:
    """Full generation pipeline for a single email (Summary + Structured Action)."""
    if model is None:
        model = load_model()

    t0 = time.time()
    summary = summarize_email(subject, body, intent=intent, priority=priority, model=model)
    action = extract_action(subject, body, intent=intent, priority=priority, model=model)
    latency_ms = (time.time() - t0) * 1000.0

    return GenerationOutput(
        summary=summary,
        action=action,
        intent_context=intent,
        priority_context=priority,
        latency_ms=latency_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local email summarization and action generation CLI")
    parser.add_argument("--subject", type=str, required=True, help="Email subject line")
    parser.add_argument("--body", type=str, required=True, help="Email body text")
    parser.add_argument("--intent", type=str, default=None, help="Optional intent classification context")
    parser.add_argument("--priority", type=str, default=None, help="Optional priority classification context")
    args = parser.parse_args()

    print("\n--- Running Local Generative AI (FLAN-T5-base) ---")
    output = process_email(
        subject=args.subject,
        body=args.body,
        intent=args.intent,
        priority=args.priority,
    )

    print("\nSUMMARY:")
    print(output.summary)
    print("\nSUGGESTED ACTION:")
    print(json.dumps(output.action.to_dict(), indent=2))
    print(f"\nLatency: {output.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
