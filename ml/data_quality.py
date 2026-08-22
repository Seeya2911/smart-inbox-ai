"""Data Quality & Leakage Detection Module for Smart Inbox INTENT Training.

Provides validation for dataset cleanliness (empty text, duplicates, label conflicts)
and strict cross-split leakage detection (exact match, normalized overlap, and near-duplicates).
The pipeline FAILS LOUDLY if data leakage is detected.
"""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

from ml.schema import CanonicalIntentExample


class DataQualityError(ValueError):
    """Raised when dataset integrity checks fail (empty text, duplicates, label conflicts)."""


class DataLeakageError(ValueError):
    """Raised when leakage is detected between train, validation, or test splits."""


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def compute_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity between two normalized strings."""
    if text1 == text2:
        return 1.0
    if len(text1) < n or len(text2) < n:
        return 1.0 if text1 == text2 else 0.0

    ngrams1 = {text1[i : i + n] for i in range(len(text1) - n + 1)}
    ngrams2 = {text2[i : i + n] for i in range(len(text2) - n + 1)}

    union = ngrams1 | ngrams2
    if not union:
        return 0.0
    intersection = ngrams1 & ngrams2
    return len(intersection) / len(union)


def check_dataset_integrity(
    examples: List[CanonicalIntentExample],
    allow_internal_duplicates: bool = False,
) -> Dict[str, int]:
    """Validate cleanliness of a dataset before splitting.

    Checks:
    - Empty text
    - Duplicate source IDs (source_dataset, source_example_id)
    - Conflicting labels (same normalized text assigned different canonical intents)
    - Duplicate text (exact and normalized) within the dataset

    Raises:
        DataQualityError: If any data quality violation is discovered.
    """
    if not examples:
        raise DataQualityError("Dataset is empty; cannot run quality checks.")

    # 1. Empty text check
    for idx, ex in enumerate(examples):
        if not ex.text or not ex.text.strip():
            raise DataQualityError(
                f"Empty text detected at index {idx} (source_dataset={ex.source_dataset!r}, ID={ex.source_example_id!r})"
            )

    # 2. Duplicate source IDs check
    seen_source_ids: Set[Tuple[str, str]] = set()
    dup_source_ids: List[Tuple[str, str]] = []
    for ex in examples:
        key = (ex.source_dataset, ex.source_example_id)
        if key in seen_source_ids:
            dup_source_ids.append(key)
        seen_source_ids.add(key)
    if dup_source_ids:
        raise DataQualityError(
            f"Duplicate source IDs detected within dataset ({len(dup_source_ids)} duplicates). "
            f"Examples: {dup_source_ids[:5]}"
        )

    # 3. Conflicting labels check (same normalized text, different canonical_intent)
    text_to_labels: Dict[str, Set[str]] = defaultdict(set)
    for ex in examples:
        norm = normalize_text(ex.text)
        text_to_labels[norm].add(ex.canonical_intent)

    conflicts = {text: labels for text, labels in text_to_labels.items() if len(labels) > 1}
    if conflicts:
        sample_conflict = next(iter(conflicts.items()))
        raise DataQualityError(
            f"Conflicting labels detected for identical normalized text ({len(conflicts)} conflict groups). "
            f"Sample text snippet: {sample_conflict[0][:80]!r} has labels {sorted(sample_conflict[1])}"
        )

    # 4. Duplicate text check
    if not allow_internal_duplicates:
        seen_norm_text: Set[str] = set()
        dup_texts: List[str] = []
        for ex in examples:
            norm = normalize_text(ex.text)
            if norm in seen_norm_text:
                dup_texts.append(norm)
            seen_norm_text.add(norm)
        if dup_texts:
            raise DataQualityError(
                f"Duplicate text detected within dataset ({len(dup_texts)} duplicates). "
                f"Sample duplicate snippet: {dup_texts[0][:80]!r}"
            )

    return {
        "total_checked": len(examples),
        "unique_source_ids": len(seen_source_ids),
        "unique_normalized_texts": len(text_to_labels),
    }


def check_split_leakage(
    train_examples: List[CanonicalIntentExample],
    val_examples: List[CanonicalIntentExample],
    test_examples: List[CanonicalIntentExample],
    near_duplicate_threshold: float = 0.85,
) -> None:
    """Check for data leakage between train, val, and test splits.

    Checks:
    - Exact text overlap between splits
    - Normalized-text overlap between splits
    - Obvious near-duplicate overlap between splits (via character n-gram similarity)

    Raises:
        DataLeakageError: FAILS LOUDLY if any split overlap or near-duplicate leakage is detected.
    """
    splits: Dict[str, List[CanonicalIntentExample]] = {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }

    split_names = list(splits.keys())
    leakage_records: List[str] = []

    # 1. Exact and normalized text overlap
    norm_text_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for split_name, examples in splits.items():
        for ex in examples:
            norm = normalize_text(ex.text)
            norm_text_map[norm][split_name].add(ex.source_example_id)

    for norm_text, split_dict in norm_text_map.items():
        if len(split_dict) > 1:
            involved_splits = sorted(split_dict.keys())
            ids_str = ", ".join(f"{s}: {sorted(split_dict[s])}" for s in involved_splits)
            leakage_records.append(
                f"[Normalized Text Overlap] Text snippet {norm_text[:80]!r} occurs across splits ({', '.join(involved_splits)}). IDs: {ids_str}"
            )

    # 2. Near-duplicate leakage check across split pairs
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1_name, s2_name = split_names[i], split_names[j]
            s1_examples, s2_examples = splits[s1_name], splits[s2_name]

            for ex1 in s1_examples:
                norm1 = normalize_text(ex1.text)
                if len(norm1) < 15:  # skip short strings for near-duplicate check
                    continue
                for ex2 in s2_examples:
                    norm2 = normalize_text(ex2.text)
                    if len(norm2) < 15 or norm1 == norm2:
                        continue

                    similarity = compute_ngram_jaccard(norm1, norm2, n=4)
                    if similarity >= near_duplicate_threshold:
                        leakage_records.append(
                            f"[Near-Duplicate Leakage] {s1_name} example ID {ex1.source_example_id!r} and {s2_name} example ID {ex2.source_example_id!r} "
                            f"have high similarity ({similarity:.3f} >= {near_duplicate_threshold}).\n"
                            f"  {s1_name}: {norm1[:80]!r}\n"
                            f"  {s2_name}: {norm2[:80]!r}"
                        )

    if leakage_records:
        error_msg = (
            f"DATA LEAKAGE DETECTED! The dataset splits contain {len(leakage_records)} overlapping or near-duplicate examples.\n"
            + "\n".join(leakage_records[:10])
        )
        if len(leakage_records) > 10:
            error_msg += f"\n... and {len(leakage_records) - 10} more leakage instances."
        raise DataLeakageError(error_msg)
