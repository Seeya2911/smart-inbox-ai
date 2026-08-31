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
) -> SuggestedAction:
    """Parse raw model text into a validated, structured SuggestedAction object.

    Post-processing rules:
    - Parses explicit model output into ALLOWED_ACTION_TYPES
    - Does NOT infer action from intent or keywords
    - Safely nulls unsupported fields
    - If raw model output is malformed or not an allowed action type, falls back to 'none'
    """
    clean_raw = (raw_model_text or "").strip().lower()
    full_text = f"{subject} {strip_email_boilerplate(body)}"

    # 1. Match explicit action type strictly from what the model generated
    action_type = "none"
    matched_explicit = False

    # Check for direct allowed type mentions in raw text
    for act in sorted(ALLOWED_ACTION_TYPES, key=lambda x: -len(x)):
        if act != "none" and (clean_raw.startswith(act) or f"action_type: {act}" in clean_raw or f"action: {act}" in clean_raw or clean_raw == act):
            action_type = act
            matched_explicit = True
            break

    if not matched_explicit:
        if any(k in clean_raw for k in ["none", "no action", "no task", "informational", "n/a"]):
            action_type = "none"
        elif clean_raw.startswith("create_calendar_event") or clean_raw.startswith("calendar"):
            action_type = "create_calendar_event"
        elif clean_raw.startswith("reply") or clean_raw.startswith("respond"):
            action_type = "reply"
        elif clean_raw.startswith("create_reminder") or clean_raw.startswith("reminder"):
            action_type = "create_reminder"
        elif clean_raw.startswith("review_document") or clean_raw.startswith("review"):
            action_type = "review_document"
        elif clean_raw.startswith("follow_up") or clean_raw.startswith("follow up"):
            action_type = "follow_up"
        elif clean_raw.startswith("contact_sender") or clean_raw.startswith("contact"):
            action_type = "contact_sender"
        elif clean_raw.startswith("create_task") or clean_raw.startswith("task"):
            action_type = "create_task"
        else:
            # Unrecognized model output -> do NOT manufacture an action
            action_type = "none"

    if action_type == "none":
        return SuggestedAction(action_type="none", title=raw_model_text.strip() if raw_model_text else None)

    # 2. Extract Title from model output
    title = raw_model_text.strip() if len(raw_model_text.strip()) > 3 else f"Action for: {subject[:100].strip()}"

    # 3. Extract dates & times only if explicitly present in email text
    date_match = _DATE_EXTRACTION_RE.search(full_text)
    due_date = date_match.group(0).strip() if date_match else None

    time_match = _TIME_EXTRACTION_RE.search(full_text)
    due_time = time_match.group(0).strip() if time_match else None

    # 4. Extract evidence snippet
    evidence_snippet = None
    clean_body = strip_email_boilerplate(body)
    if clean_body:
        first_sentence = clean_body.split(".")[0].strip()
        if len(first_sentence) > 10:
            evidence_snippet = first_sentence[:200]

    return SuggestedAction(
        action_type=action_type,
        title=title[:150],
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
) -> Tuple[str, str]:
    """Generate summary of email using local FLAN-T5 model.

    Returns:
        (clean_summary, raw_model_summary)
    """
    if model is None:
        model = load_model()

    clean_body = strip_email_boilerplate(body or "")
    if len(clean_body) < 15 and len(subject or "") < 15:
        res = f"{subject.strip()}: {clean_body.strip()}".strip(" :")
        return res, res

    prompt = build_summarization_prompt(subject, body, intent=intent, priority=priority)
    raw_summary = model.generate(
        prompt,
        max_new_tokens=96,
        min_length=8,
        num_beams=2,
        length_penalty=1.0,
    )

    clean_summary = raw_summary.strip()
    if not clean_summary:
        clean_summary = f"{subject.strip()}. {clean_body[:100].strip()}..."
    return clean_summary, raw_summary


def extract_action(
    subject: str,
    body: str,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    model: Optional[LocalGenerationModel] = None,
) -> Tuple[SuggestedAction, str]:
    """Extract action from email using local FLAN-T5 model without heuristic keyword replacement.

    Returns:
        (parsed_action, raw_model_action_text)
    """
    if model is None:
        model = load_model()

    prompt = build_action_extraction_prompt(subject, body, intent=intent, priority=priority)
    raw_action_text = model.generate(
        prompt,
        max_new_tokens=64,
        min_length=2,
        num_beams=2,
    )

    structured_action = parse_action_output(raw_action_text, subject, body)
    return structured_action, raw_action_text


def process_email(
    subject: str,
    body: str,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
    model: Optional[LocalGenerationModel] = None,
) -> GenerationOutput:
    """Full generation pipeline preserving raw model output alongside parsed output."""
    if model is None:
        model = load_model()

    t0 = time.time()
    summary, raw_summary = summarize_email(subject, body, intent=intent, priority=priority, model=model)
    action, raw_action = extract_action(subject, body, intent=intent, priority=priority, model=model)
    latency_ms = (time.time() - t0) * 1000.0

    return GenerationOutput(
        summary=summary,
        action=action,
        raw_model_summary=raw_summary,
        raw_model_action=raw_action,
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
