"""Deterministic group-aware stratified dataset splitting.

Examples are isolated by their source group and by near-duplicate text clusters
before stratified bin-packing. This prevents small wording variations from being
placed in different evaluation splits and leaking information across the benchmark.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ml.data_quality import check_split_leakage, compute_ngram_jaccard, normalize_text
from ml.schema import CanonicalIntentExample


NEAR_DUPLICATE_THRESHOLD = 0.85
WORD_JACCARD_THRESHOLD = 0.80


def _word_jaccard(text1: str, text2: str) -> float:
    """Return token-set Jaccard similarity for two normalized strings."""
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def _near_duplicate_group_ids(
    examples: List[CanonicalIntentExample],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> Dict[int, int]:
    """Return deterministic connected-component IDs for source groups and near duplicates.

    The component is the indivisible unit used by the splitter. Two examples are
    clustered when either the existing character 4-gram leakage metric reaches the
    supplied threshold or their token-set Jaccard is at least 0.80. The token signal
    catches small lexical edits such as ``email`` -> ``emails`` and inserted prefixes
    such as ``hi``/``olly`` while avoiding obvious substitutions such as ``sam`` ->
    ``alex``.
    """
    if not examples:
        return {}
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"Near-duplicate threshold must be in (0, 1]; got {threshold}")

    parent = list(range(len(examples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    # Preserve the existing source-group isolation guarantee.
    source_groups: Dict[str, List[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        source_groups[example.source_group_id or example.source_example_id].append(index)
    for indices in source_groups.values():
        first = indices[0]
        for index in indices[1:]:
            union(first, index)

    normalized = [normalize_text(example.text) for example in examples]

    # Exact normalized duplicates are cheap to identify and should always share a group.
    first_by_text: Dict[str, int] = {}
    for index, text in enumerate(normalized):
        previous = first_by_text.get(text)
        if previous is None:
            first_by_text[text] = index
        else:
            union(previous, index)

    # Build a token index so the near-duplicate search avoids comparing every pair
    # that has no meaningful lexical overlap. The dataset is currently small enough
    # that candidate expansion remains deterministic and inexpensive in CI.
    token_to_indices: Dict[str, Set[int]] = defaultdict(set)
    token_sets: List[Set[str]] = []
    ngram_sets: List[Set[str]] = []
    for index, text in enumerate(normalized):
        tokens = set(text.split())
        token_sets.append(tokens)
        for token in tokens:
            token_to_indices[token].add(index)
        ngram_sets.append({text[pos : pos + 4] for pos in range(max(0, len(text) - 3))})

    for index, text in enumerate(normalized):
        if len(text) < 15:
            continue

        # Compare against examples sharing at least one token. Generic tokens such as
        # "email" can produce many candidates, but the final similarity tests remain
        # strict and deterministic.
        candidates: Set[int] = set()
        for token in token_sets[index]:
            candidates.update(token_to_indices[token])
        candidates.discard(index)

        for other in sorted(candidates):
            if other <= index or len(normalized[other]) < 15:
                continue

            # A length-ratio guard prevents short utterances from being absorbed into
            # much longer utterances merely because they share common words.
            len1 = len(normalized[index])
            len2 = len(normalized[other])
            if min(len1, len2) / max(len1, len2) < 0.70:
                continue

            word_similarity = _word_jaccard(normalized[index], normalized[other])
            char_similarity = compute_ngram_jaccard(normalized[index], normalized[other], n=4)

            if word_similarity >= WORD_JACCARD_THRESHOLD or char_similarity >= threshold:
                union(index, other)

    # Convert roots to stable compact IDs based on first occurrence.
    root_to_component: Dict[int, int] = {}
    component_ids: Dict[int, int] = {}
    next_component = 0
    for index in range(len(examples)):
        root = find(index)
        if root not in root_to_component:
            root_to_component[root] = next_component
            next_component += 1
        component_ids[index] = root_to_component[root]
    return component_ids


def split_intent_dataset(
    examples: List[CanonicalIntentExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    require_multi_class: bool = True,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> Tuple[List[CanonicalIntentExample], List[CanonicalIntentExample], List[CanonicalIntentExample]]:
    """Split dataset deterministically while isolating source groups and near duplicates."""
    if not examples:
        raise ValueError("Cannot split an empty dataset.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    overall_class_counts: Dict[str, int] = defaultdict(int)
    for example in examples:
        overall_class_counts[example.canonical_intent] += 1

    distinct_classes = sorted(overall_class_counts)
    if require_multi_class and len(distinct_classes) < 2:
        raise ValueError(
            f"Dataset has only {len(distinct_classes)} class ({distinct_classes}); "
            "multi-class intent training and evaluation requires at least 2 distinct intent classes."
        )

    # A connected component combines source-group isolation with near-duplicate
    # isolation, so no known leakage pair can be split apart later.
    component_ids = _near_duplicate_group_ids(examples, threshold=near_duplicate_threshold)
    group_to_examples: Dict[str, List[CanonicalIntentExample]] = defaultdict(list)
    for index, example in enumerate(examples):
        group_to_examples[f"cluster-{component_ids[index]}"] .append(example)

    group_ids = sorted(group_to_examples)
    group_class_counts: Dict[str, Dict[str, int]] = {}
    for group_id, group_examples in group_to_examples.items():
        counts: Dict[str, int] = defaultdict(int)
        for example in group_examples:
            counts[example.canonical_intent] += 1
        group_class_counts[group_id] = counts

    split_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    active_splits = [split for split, ratio in split_ratios.items() if ratio > 0]
    target_counts = {
        split: {intent: overall_class_counts[intent] * split_ratios[split] for intent in distinct_classes}
        for split in active_splits
    }
    allocated_counts = {
        split: {intent: 0.0 for intent in distinct_classes} for split in active_splits
    }

    rng = np.random.RandomState(seed)

    def group_rarity_key(group_id: str) -> int:
        return min(overall_class_counts[intent] for intent in group_class_counts[group_id])

    tier_groups: Dict[int, List[str]] = defaultdict(list)
    for group_id in group_ids:
        tier_groups[group_rarity_key(group_id)].append(group_id)

    ordered_groups: List[str] = []
    for tier in sorted(tier_groups):
        candidates = tier_groups[tier]
        rng.shuffle(candidates)
        ordered_groups.extend(candidates)

    group_split_assignment: Dict[str, str] = {}
    for group_id in ordered_groups:
        group_counts = group_class_counts[group_id]
        best_split: Optional[str] = None
        best_score = -float("inf")

        for split in active_splits:
            deficit_score = 0.0
            for intent, count in group_counts.items():
                target = target_counts[split][intent]
                current = allocated_counts[split][intent]
                deficit = target - current
                deficit_score += (deficit / target) * count if target > 0 else deficit * count
            if deficit_score > best_score:
                best_score = deficit_score
                best_split = split

        assert best_split is not None
        group_split_assignment[group_id] = best_split
        for intent, count in group_counts.items():
            allocated_counts[best_split][intent] += count

    train_groups = {group for group, split in group_split_assignment.items() if split == "train"}
    val_groups = {group for group, split in group_split_assignment.items() if split == "val"}
    test_groups = {group for group, split in group_split_assignment.items() if split == "test"}

    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("Group isolation violation: a leakage-aware group appeared in multiple splits.")

    train_examples = [example for group in sorted(train_groups) for example in group_to_examples[group]]
    val_examples = [example for group in sorted(val_groups) for example in group_to_examples[group]]
    test_examples = [example for group in sorted(test_groups) for example in group_to_examples[group]]

    if not train_examples:
        raise ValueError(
            f"Stratified group splitting produced an empty train split (val={len(val_examples)}, test={len(test_examples)}). "
            "Dataset lacks enough groups to form a training split."
        )
    if val_ratio > 0 and not val_examples:
        raise ValueError(
            f"Stratified group splitting produced an empty validation split (train={len(train_examples)}, test={len(test_examples)}). "
            "Dataset lacks enough groups for validation."
        )
    if test_ratio > 0 and not test_examples:
        raise ValueError(
            f"Stratified group splitting produced an empty test split (train={len(train_examples)}, val={len(val_examples)}). "
            "Dataset lacks enough groups for test split."
        )

    train_classes = {example.canonical_intent for example in train_examples}
    test_classes = {example.canonical_intent for example in test_examples} if test_examples else set()
    missing_train_classes = set(distinct_classes) - train_classes
    if missing_train_classes:
        raise ValueError(
            f"Stratified group splitting failed: class(es) {sorted(missing_train_classes)} missing from train split. "
            "Dataset has insufficient groups per class to achieve stratified split."
        )
    if require_multi_class and len(test_classes) < 2:
        raise ValueError(
            f"Invalid single-class test split produced (test classes: {sorted(test_classes)}). "
            "Multi-class intent evaluation requires at least 2 distinct classes in test split."
        )

    check_split_leakage(train_examples, val_examples, test_examples)
    return train_examples, val_examples, test_examples
