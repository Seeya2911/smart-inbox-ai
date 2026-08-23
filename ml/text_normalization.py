"""Shared text normalization for intent-data preparation and validation."""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
