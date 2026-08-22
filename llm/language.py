"""Language detection and lightweight language-specific validation.

This module is intentionally not a semantic classifier. It identifies the likely
language before LLM analysis and provides conservative validation signals. The
original email text is preserved for the LLM so grammar and context are not
removed by traditional preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en", "de", "fr", "es"}

# A small set of high-signal function words is used only as a sanity check.
# It is not used to classify intent, urgency, priority, or sentiment.
_LANGUAGE_MARKERS = {
    "en": {"the", "and", "please", "with", "your"},
    "de": {"der", "die", "das", "und", "bitte", "mit", "ihre"},
    "fr": {"le", "la", "les", "et", "avec", "veuillez", "votre"},
    "es": {"el", "la", "los", "y", "por", "favor", "su"},
}


@dataclass(frozen=True)
class LanguageResult:
    language: str
    confidence: float
    supported: bool
    marker_score: float


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ÿ]+", text.lower(), flags=re.UNICODE))


def detect_language(text: str) -> LanguageResult:
    """Detect one of the supported languages, conservatively.

    Very short text is treated as uncertain because language detectors are
    unreliable when there is insufficient linguistic context.
    """
    clean = " ".join(text.split())
    if len(clean) < 20:
        return LanguageResult("unknown", 0.0, False, 0.0)

    try:
        candidates = detect_langs(clean)
    except Exception:
        return LanguageResult("unknown", 0.0, False, 0.0)

    best_supported = next((c for c in candidates if c.lang in SUPPORTED_LANGUAGES), None)
    if best_supported is None:
        return LanguageResult("unknown", 0.0, False, 0.0)

    tokens = _tokens(clean)
    markers = _LANGUAGE_MARKERS[best_supported.lang]
    marker_score = len(tokens & markers) / max(1, len(markers))
    return LanguageResult(
        language=best_supported.lang,
        confidence=float(best_supported.prob),
        supported=True,
        marker_score=marker_score,
    )


def validate_language(text: str, expected: str) -> LanguageResult:
    """Validate an expected language against independent detection signals."""
    result = detect_language(text)
    if expected not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported expected language: {expected}")
    return result
