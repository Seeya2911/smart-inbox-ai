"""Email Boilerplate Stripping and Near-Duplicate Deduplication Module.

Strips quoted reply chains (> ...), forward headers, signatures, and legal disclaimers
before computing exact or fuzzy near-duplicate hashes.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from ml.schema import CanonicalEmailExample

# Forward header patterns
FORWARD_HEADER_PATTERNS = [
    r"-+\s*Forwarded message\s*-+",
    r"-+\s*Original Message\s*-+",
    r"From:\s*.*?\nSent:\s*.*?\nTo:\s*.*?\nSubject:\s*.*",
    r"Begin forwarded message:",
]

# Disclaimer patterns
DISCLAIMER_PATTERNS = [
    r"this email and any files transmitted with it are confidential.*",
    r"if you have received this email in error please notify.*",
    r"the contents of this email message and any attachments are intended solely.*",
]

# Common sign-off phrases
SIGN_OFF_PATTERNS = [
    r"\n\s*(?:thanks|best regards|kind regards|cheers|sincerely|warm regards|regards|yours truly|best),?\s*\n.*$",
]


def strip_email_boilerplate(text: str) -> str:
    """Strip quoted reply chains (> ...), forward headers, signatures, and legal disclaimers."""
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    # Remove forward headers
    for pat in FORWARD_HEADER_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove quoted reply lines (lines starting with >)
    lines = cleaned.splitlines()
    filtered_lines = [line for line in lines if not line.strip().startswith(">")]
    cleaned = "\n".join(filtered_lines)

    # Remove common disclaimers
    for pat in DISCLAIMER_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove trailing signature blocks
    for pat in SIGN_OFF_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Normalize unicode and collapse whitespace
    cleaned = unicodedata.normalize("NFKC", cleaned).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def compute_content_hash(text: str) -> str:
    """Compute MD5 hash of boilerplate-stripped email content."""
    clean = strip_email_boilerplate(text)
    return hashlib.md5(clean.encode("utf-8")).hexdigest()


def ngram_jaccard_similarity(t1: str, t2: str, n: int = 4) -> float:
    """Compute character n-gram Jaccard similarity between two strings."""
    if t1 == t2:
        return 1.0
    if len(t1) < n or len(t2) < n:
        return 1.0 if t1 == t2 else 0.0
    ng1 = {t1[i : i + n] for i in range(len(t1) - n + 1)}
    ng2 = {t2[i : i + n] for i in range(len(t2) - n + 1)}
    union = ng1 | ng2
    if not union:
        return 0.0
    return len(ng1 & ng2) / len(union)


def is_superset_duplicate(t1: str, t2: str, min_len: int = 30) -> bool:
    """Check if one boilerplate-stripped body is an exact substring/superset of another."""
    if len(t1) < min_len or len(t2) < min_len:
        return False
    return (t1 in t2) or (t2 in t1)


def deduplicate_dataset(
    examples: List[CanonicalEmailExample],
    sim_threshold: float = 0.85,
) -> Tuple[List[CanonicalEmailExample], Dict[str, Any]]:
    """Deduplicate canonical email examples based on boilerplate-stripped content.

    Returns:
        (deduplicated_examples, stats_dict)
    """
    if not examples:
        return [], {"total_input": 0, "unique_output": 0, "exact_duplicates": 0, "near_duplicates": 0}

    deduped: List[CanonicalEmailExample] = []
    seen_hashes: Set[str] = set()
    seen_clean_texts: List[str] = []

    exact_dups = 0
    near_dups = 0

    for ex in examples:
        clean_body = strip_email_boilerplate(ex.full_text)
        content_hash = hashlib.md5(clean_body.encode("utf-8")).hexdigest()

        if content_hash in seen_hashes:
            exact_dups += 1
            continue

        # Check near-duplicates and superset containment
        is_dup = False
        if len(clean_body) >= 20:
            for prev_text in seen_clean_texts:
                if len(prev_text) < 20:
                    continue
                # Superset / substring check
                if is_superset_duplicate(clean_body, prev_text):
                    is_dup = True
                    break
                # Fuzzy n-gram Jaccard check
                if abs(len(clean_body) - len(prev_text)) < len(clean_body) * 0.4:
                    if ngram_jaccard_similarity(clean_body, prev_text, n=4) >= sim_threshold:
                        is_dup = True
                        break

        if is_dup:
            near_dups += 1
            continue

        seen_hashes.add(content_hash)
        seen_clean_texts.append(clean_body)
        deduped.append(ex)

    stats = {
        "total_input": len(examples),
        "unique_output": len(deduped),
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "total_removed": exact_dups + near_dups,
    }

    return deduped, stats
