import re
from typing import Dict, List


def clean_text_for_summary(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "[Link]", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_key_sentences(text: str, max_sentences: int = 3) -> List[str]:
    clean_text = clean_text_for_summary(text)
    if not clean_text:
        return []
    sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    important_words = {"urgent", "important", "deadline", "meeting", "please", "need", "must", "should"}
    scored = []
    for sentence in sentences:
        score = len(sentence.split()) * 0.1
        score += sum(2 for word in important_words if word in sentence.lower())
        scored.append((sentence, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [sentence for sentence, _ in scored[:max_sentences]]


def generate_email_summary(email: Dict, max_length: int = 150) -> str:
    subject = email.get("subject", "No Subject")
    body = clean_text_for_summary(email.get("body", ""))
    if not body:
        return f"Subject: {subject}"
    summary = " ".join(extract_key_sentences(body, max_sentences=2)) or body
    return summary if len(summary) <= max_length else summary[: max_length - 3] + "..."


def format_email_display(email: Dict) -> Dict:
    subject = email.get("subject", "").strip()
    if not subject:
        words = clean_text_for_summary(email.get("body", "")).split()[:5]
        email["subject"] = " ".join(words) or "No Subject"
    email["summary"] = generate_email_summary(email)
    return email


class EmailSummarizer:
    """Backward-compatible wrapper around the legacy extractive baseline."""

    def summarize(self, email: Dict, max_length: int = 150) -> str:
        return generate_email_summary(email, max_length=max_length)
