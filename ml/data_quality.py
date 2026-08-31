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


# ---------------------------------------------------------------------------
# Multi-output (CanonicalEmailExample) quality + leakage checks
# ---------------------------------------------------------------------------

from ml.schema import ALLOWED_PRIORITIES, CanonicalEmailExample  # noqa: E402


def check_email_dataset_integrity(
    examples: List["CanonicalEmailExample"],
    allow_internal_duplicates: bool = False,
) -> Dict[str, int]:
    """Validate cleanliness of a multi-output (intent + priority) dataset.

    Checks performed:
    - Empty dataset
    - Empty body
    - Missing or invalid intent (must be one of 11 ALLOWED_INTENTS)
    - Missing or invalid priority (must be one of 3 ALLOWED_PRIORITIES)
    - Duplicate example IDs
    - Conflicting intent labels for identical IDs
    - Conflicting priority labels for identical IDs
    - Duplicate normalized text (optional)

    Raises DataQualityError on any violation.
    Returns summary statistics dict on success.
    """
    if not examples:
        raise DataQualityError("Dataset is empty; cannot run quality checks.")

    # 1. Per-example field checks
    for idx, ex in enumerate(examples):
        body = ex.body.strip() if isinstance(ex.body, str) else ""
        if not body:
            raise DataQualityError(
                f"Empty body at index {idx} (id={ex.id!r})"
            )
        if ex.intent not in ALLOWED_INTENTS:
            raise DataQualityError(
                f"Invalid intent {ex.intent!r} at index {idx} (id={ex.id!r}). "
                f"Must be one of {sorted(ALLOWED_INTENTS)}"
            )
        if ex.priority not in ALLOWED_PRIORITIES:
            raise DataQualityError(
                f"Invalid priority {ex.priority!r} at index {idx} (id={ex.id!r}). "
                f"Must be one of {sorted(ALLOWED_PRIORITIES)}"
            )

    # 2. Duplicate ID checks
    id_to_intents: Dict[str, Set[str]] = defaultdict(set)
    id_to_priorities: Dict[str, Set[str]] = defaultdict(set)
    id_count: Dict[str, int] = defaultdict(int)
    for ex in examples:
        id_to_intents[ex.id].add(ex.intent)
        id_to_priorities[ex.id].add(ex.priority)
        id_count[ex.id] += 1

    # Conflicting labels for same ID
    conflict_intent = {eid: v for eid, v in id_to_intents.items() if len(v) > 1}
    if conflict_intent:
        sample_eid, sample_intents = next(iter(conflict_intent.items()))
        raise DataQualityError(
            f"Conflicting intent labels for {len(conflict_intent)} IDs. "
            f"Example id={sample_eid!r} has intents: {sorted(sample_intents)}"
        )
    conflict_priority = {eid: v for eid, v in id_to_priorities.items() if len(v) > 1}
    if conflict_priority:
        sample_eid, sample_pris = next(iter(conflict_priority.items()))
        raise DataQualityError(
            f"Conflicting priority labels for {len(conflict_priority)} IDs. "
            f"Example id={sample_eid!r} has priorities: {sorted(sample_pris)}"
        )

    # Duplicate IDs (same ID more than once, even with same label)
    dup_ids = {eid: cnt for eid, cnt in id_count.items() if cnt > 1}
    if dup_ids:
        raise DataQualityError(
            f"Duplicate IDs found: {len(dup_ids)} IDs appear more than once. "
            f"Examples: {list(dup_ids.items())[:5]}"
        )

    # 3. Duplicate normalized text
    if not allow_internal_duplicates:
        seen_norm: Set[str] = set()
        dup_texts: List[str] = []
        for ex in examples:
            norm = normalize_text(ex.full_text)
            if norm in seen_norm:
                dup_texts.append(norm)
            seen_norm.add(norm)
        if dup_texts:
            raise DataQualityError(
                f"Duplicate normalized text found ({len(dup_texts)} duplicates). "
                f"Sample: {dup_texts[0][:80]!r}"
            )

    return {
        "total_checked": len(examples),
        "unique_ids": len(id_count),
        "unique_normalized_texts": len({normalize_text(ex.full_text) for ex in examples}),
    }


def check_split_leakage_email(
    train_examples: List["CanonicalEmailExample"],
    val_examples: List["CanonicalEmailExample"],
    test_examples: List["CanonicalEmailExample"],
    near_duplicate_threshold: float = 0.85,
) -> None:
    """Check for data leakage between train, val, and test splits for multi-output pipeline.

    Checks:
    - Exact ID overlap
    - Normalized-text overlap
    - Source-group overlap
    - Near-duplicate overlap (character n-gram Jaccard)

    Raises DataLeakageError if any overlap is found.
    """
    splits: Dict[str, List["CanonicalEmailExample"]] = {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }
    split_names = list(splits.keys())
    leakage_records: List[str] = []

    # 1. ID overlap
    split_ids: Dict[str, Set[str]] = {
        name: {ex.id for ex in exs} for name, exs in splits.items()
    }
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = split_ids[s1] & split_ids[s2]
            if overlap:
                leakage_records.append(
                    f"[ID Leakage] {len(overlap)} IDs overlap between '{s1}' and '{s2}'. "
                    f"Sample: {sorted(overlap)[:5]}"
                )

    # 2. Source-group overlap
    split_groups: Dict[str, Set[str]] = {
        name: {ex.source_group_id or ex.id for ex in exs}
        for name, exs in splits.items()
    }
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = split_groups[s1] & split_groups[s2]
            if overlap:
                leakage_records.append(
                    f"[Source-Group Leakage] {len(overlap)} group IDs overlap between '{s1}' and '{s2}'. "
                    f"Sample: {sorted(overlap)[:5]}"
                )

    # 3. Normalized text overlap
    norm_text_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for split_name, exs in splits.items():
        for ex in exs:
            norm = normalize_text(ex.full_text)
            norm_text_map[norm][split_name].add(ex.id)

    for norm_text, split_dict in norm_text_map.items():
        if len(split_dict) > 1:
            involved = sorted(split_dict.keys())
            ids_str = ", ".join(f"{s}: {sorted(split_dict[s])}" for s in involved)
            leakage_records.append(
                f"[Normalized Text Overlap] Text {norm_text[:60]!r} spans splits "
                f"({', '.join(involved)}). IDs: {ids_str}"
            )

    # 4. Near-duplicate check (across splits)
    split_cache = {
        name: [
            (ex.id, normalize_text(ex.full_text), len(normalize_text(ex.full_text)), set(normalize_text(ex.full_text).split()))
            for ex in exs
        ]
        for name, exs in splits.items()
    }
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1_name, s2_name = split_names[i], split_names[j]
            s1_cached = [item for item in split_cache[s1_name] if item[2] >= 15]
            s2_cached = [item for item in split_cache[s2_name] if item[2] >= 15]
            for id1, norm1, len1, w1 in s1_cached:
                for id2, norm2, len2, w2 in s2_cached:
                    if norm1 == norm2:
                        continue
                    if min(len1, len2) / max(len1, len2) < 0.70:
                        continue
                    w_inter = len(w1 & w2)
                    if not w_inter:
                        continue
                    w_sim = w_inter / len(w1 | w2)
                    if w_sim < 0.35:
                        continue
                    sim = compute_ngram_jaccard(norm1, norm2, n=4)
                    if sim >= near_duplicate_threshold:
                        leakage_records.append(
                            f"[Near-Dup Leakage] {s1_name} id={id1!r} vs {s2_name} id={id2!r} "
                            f"similarity={sim:.3f}"
                        )

    if leakage_records:
        error_msg = (
            f"DATA LEAKAGE DETECTED in email splits! "
            f"{len(leakage_records)} overlap(s):\n"
            + "\n".join(leakage_records[:10])
        )
        if len(leakage_records) > 10:
            error_msg += f"\n... and {len(leakage_records) - 10} more."
        raise DataLeakageError(error_msg)

