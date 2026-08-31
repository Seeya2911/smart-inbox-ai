"""Prompt templates and builders for local email generation and extraction."""
from __future__ import annotations

from typing import Optional

from ml.deduplication import strip_email_boilerplate


def clean_email_text(subject: str, body: str, max_chars: int = 1500) -> str:
    """Format and truncate email text cleanly without signatures/boilerplate."""
    subj = (subject or "").strip()
    if subj.lower() in ("nan", "none", "null"):
        subj = ""
    clean_b = strip_email_boilerplate((body or "").strip())
    if clean_b.lower() in ("nan", "none", "null"):
        clean_b = ""
    if len(clean_b) > max_chars:
        clean_b = clean_b[:max_chars] + "... [truncated]"
    if subj:
        return f"Subject: {subj}\nBody: {clean_b}"
    return f"Body: {clean_b}"


def build_summarization_prompt(
    subject: str,
    body: str,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
) -> str:
    """Construct prompt for concise, factual email summarization."""
    formatted_text = clean_email_text(subject, body)
    context_prefix = ""
    if intent or priority:
        parts = []
        if intent:
            parts.append(f"Category: {intent}")
        if priority:
            parts.append(f"Priority: {priority}")
        context_prefix = f"[{', '.join(parts)}]\n"

    prompt = (
        f"Summarize the following email concisely and factually in 1-2 sentences. "
        f"Preserve key dates, names, amounts, and requested actions:\n\n"
        f"{context_prefix}"
        f"{formatted_text}\n\n"
        f"Summary:"
    )
    return prompt


def build_action_extraction_prompt(
    subject: str,
    body: str,
    intent: Optional[str] = None,
    priority: Optional[str] = None,
) -> str:
    """Construct prompt for extracting structured actionable tasks from email."""
    formatted_text = clean_email_text(subject, body)
    context_prefix = ""
    if intent or priority:
        parts = []
        if intent:
            parts.append(f"Category: {intent}")
        if priority:
            parts.append(f"Priority: {priority}")
        context_prefix = f"[{', '.join(parts)}]\n"

    prompt = (
        f"Extract any required task or next action from this email. "
        f"Choose action type from: reply, create_task, create_reminder, create_calendar_event, review_document, contact_sender, follow_up, none.\n\n"
        f"{context_prefix}"
        f"{formatted_text}\n\n"
        f"Task Action Type and Details:"
    )
    return prompt
