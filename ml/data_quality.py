"""Data Quality & Leakage Detection Module for Smart Inbox INTENT Training.

Provides validation for dataset cleanliness (empty text, missing labels, duplicate source IDs,
conflicting labels for identical source IDs/text) and strict cross-split leakage detection
(exact match, normalized overlap, source-group overlap, and near-duplicates).

The pipeline FAILS LOUDLY if data leakage or data quality violations are detected.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ml.schema import ALLOWED_INTENTS, CanonicalIntentExample
from ml.text_normalization import normalize_text


class DataQualityError(ValueError):
    """Raised when dataset integrity checks fail (empty text, duplicates, label conflicts)."""


class DataLeakageError(ValueError):
    """Raised when leakage is detected between train, validation, or test splits."""


def compute_ngram_jaccard(text1: str, text2: str, n: int = 4) -> float:
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
    """Validate cleanliness of a dataset before splitting or model fitting.

    Checks:
    - Empty dataset
    - Empty text
    - Missing or invalid canonical intent labels
    - Duplicate source IDs (source_dataset, source_example_id)
    - Conflicting labels for the same source ID or same normalized text
    - Duplicate text (exact and normalized) within the dataset

    Raises:
        DataQualityError: If any data quality violation is discovered.
    """
    if not examples:
        raise DataQualityError("Dataset is empty; cannot run quality checks.")

    # 1. Empty text & Missing label checks
    for idx, ex in enumerate(examples):
        if not ex.text or not ex.text.strip():
            raise DataQualityError(
                f"Empty text detected at index {idx} (source_dataset={ex.source_dataset!r}, ID={ex.source_example_id!r})"
            )
        if not ex.canonical_intent or ex.canonical_intent not in ALLOWED_INTENTS:
            raise DataQualityError(
                f"Missing or invalid canonical_intent label {ex.canonical_intent!r} at index {idx} "
                f"(source_dataset={ex.source_dataset!r}, ID={ex.source_example_id!r})"
            )

    # 2. Duplicate source IDs & Conflicting labels for same source ID check
    source_id_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for ex in examples:
        key = (ex.source_dataset, ex.source_example_id)
        source_id_map[key].add(ex.canonical_intent)

    dup_source_ids = [key for key, intents in source_id_map.items() if len(intents) > 1 or sum(1 for e in examples if (e.source_dataset, e.source_example_id) == key) > 1]
    conflict_source_ids = {key: intents for key, intents in source_id_map.items() if len(intents) > 1}

    if conflict_source_ids:
        sample_key, sample_intents = next(iter(conflict_source_ids.items()))
        raise DataQualityError(
            f"Conflicting labels detected for identical source ID ({len(conflict_source_ids)} conflict groups). "
            f"Source dataset={sample_key[0]!r}, ID={sample_key[1]!r} has conflicting labels: {sorted(sample_intents)}"
        )

    if dup_source_ids:
        raise DataQualityError(
            f"Duplicate source IDs detected within dataset ({len(dup_source_ids)} duplicate source IDs). "
            f"Examples: {dup_source_ids[:5]}"
        )

    # 3. Conflicting labels check for identical normalized text
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

    # 4. Duplicate text check within dataset
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
        "unique_source_ids": len(source_id_map),
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
    - Duplicate source IDs across splits
    - Source-group overlap across splits
    - Obvious near-duplicate overlap between splits (via character n-gram Jaccard similarity)

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

    # 1. Source Group Overlap Check across splits
    split_groups: Dict[str, Set[str]] = defaultdict(set)
    for split_name, examples in splits.items():
        for ex in examples:
            gid = ex.source_group_id if ex.source_group_id else ex.source_example_id
            split_groups[split_name].add(gid)

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap_gids = split_groups[s1] & split_groups[s2]
            if overlap_gids:
                leakage_records.append(
                    f"[Source-Group Leakage] {len(overlap_gids)} group ID(s) occur across splits '{s1}' and '{s2}'. "
                    f"Sample overlapping group IDs: {sorted(list(overlap_gids))[:5]}"
                )

    # 2. Duplicate Source ID Check across splits
    split_source_ids: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for split_name, examples in splits.items():
        for ex in examples:
            split_source_ids[split_name].add((ex.source_dataset, ex.source_example_id))

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap_ids = split_source_ids[s1] & split_source_ids[s2]
            if overlap_ids:
                leakage_records.append(
                    f"[Duplicate Source ID Leakage] {len(overlap_ids)} source ID(s) occur across splits '{s1}' and '{s2}'. "
                    f"Sample overlapping IDs: {sorted(list(overlap_ids))[:5]}"
                )

    # 3. Exact and Normalized Text Overlap Check across splits
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

    # 4. Near-duplicate leakage check across split pairs
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
