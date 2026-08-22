"""Deterministic Group-Aware Stratified Dataset Splitting Module.

Algorithm:
1. Groups all examples by `source_group_id` (fallback to `source_example_id`).
2. Guarantees that ALL examples sharing a group ID remain in exactly ONE split.
3. Performs group-level stratified bin-packing using a fixed random seed to preserve
   class distributions across train, validation, and test splits.
4. Validates class coverage: Ensures every split contains required class representation
   and rejects single-class or empty test splits when multi-class evaluation is required.

Limitations:
If a dataset contains very few groups per class or extreme class imbalance,
group-aware stratification cannot mathematically allocate every class to every split
without violating group isolation. In such cases, the splitter FAILS CLEARLY with a
ValueError rather than producing a silent, invalid benchmark.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from ml.data_quality import check_split_leakage
from ml.schema import CanonicalIntentExample


def split_intent_dataset(
    examples: List[CanonicalIntentExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    require_multi_class: bool = True,
) -> Tuple[List[CanonicalIntentExample], List[CanonicalIntentExample], List[CanonicalIntentExample]]:
    """Split dataset deterministically into train, validation, and test splits.

    Guarantees:
    - Group isolation: No source_group_id appears in more than one split.
    - Determinism: Fixed random seed produces identical splits across runs.
    - Class stratification: Preserves class distribution across splits as much as possible.
    - Strict validation: Fails clearly if dataset lacks sufficient groups or class coverage.

    Args:
        examples: Clean list of CanonicalIntentExample instances.
        train_ratio: Ratio of data for training split (e.g., 0.70).
        val_ratio: Ratio of data for validation split (e.g., 0.15).
        test_ratio: Ratio of data for test split (e.g., 0.15).
        seed: Fixed random seed for reproducibility.
        require_multi_class: If True, requires test split to contain >= 2 intent classes.

    Returns:
        (train_examples, val_examples, test_examples)
    """
    if not examples:
        raise ValueError("Cannot split an empty dataset.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}")

    overall_class_counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        overall_class_counts[ex.canonical_intent] += 1

    distinct_classes = sorted(list(overall_class_counts.keys()))
    if require_multi_class and len(distinct_classes) < 2:
        raise ValueError(
            f"Dataset has only {len(distinct_classes)} class ({distinct_classes}); "
            "multi-class intent training and evaluation requires at least 2 distinct intent classes."
        )

    # 1. Group examples by source_group_id (fallback to source_example_id)
    group_to_examples: Dict[str, List[CanonicalIntentExample]] = defaultdict(list)
    for ex in examples:
        gid = ex.source_group_id if ex.source_group_id else ex.source_example_id
        group_to_examples[gid].append(ex)

    group_ids = sorted(list(group_to_examples.keys()))

    # Calculate class counts per group
    group_class_counts: Dict[str, Dict[str, int]] = {}
    for gid, grp_exs in group_to_examples.items():
        counts: Dict[str, int] = defaultdict(int)
        for ex in grp_exs:
            counts[ex.canonical_intent] += 1
        group_class_counts[gid] = counts

    # 2. Target class counts per split
    split_ratios: Dict[str, float] = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    active_splits = [s for s, r in split_ratios.items() if r > 0]

    target_counts: Dict[str, Dict[str, float]] = {
        s: {c: overall_class_counts[c] * split_ratios[s] for c in distinct_classes} for s in active_splits
    }
    allocated_counts: Dict[str, Dict[str, float]] = {
        s: {c: 0.0 for c in distinct_classes} for s in active_splits
    }

    # 3. Deterministic Group Order: Prioritize groups with rarer classes, then shuffle with seed
    rng = np.random.RandomState(seed)

    def group_rarity_key(gid: str) -> float:
        # Minimum overall class frequency present in group
        min_freq = min(overall_class_counts[c] for c in group_class_counts[gid].keys())
        return min_freq

    # Bucket groups by rarity tier, shuffle within tier
    tier_groups: Dict[float, List[str]] = defaultdict(list)
    for gid in group_ids:
        tier = group_rarity_key(gid)
        tier_groups[tier].append(gid)

    ordered_groups: List[str] = []
    for tier in sorted(tier_groups.keys()):
        g_list = tier_groups[tier]
        rng.shuffle(g_list)
        ordered_groups.extend(g_list)

    # 4. Greedy Stratified Bin-Packing of Groups into Splits
    group_split_assignment: Dict[str, str] = {}

    for gid in ordered_groups:
        grp_counts = group_class_counts[gid]

        best_split: Optional[str] = None
        best_score = -float("inf")

        for s in active_splits:
            # Deficit score: how much this split needs the classes in group gid
            deficit_score = 0.0
            for c, cnt in grp_counts.items():
                target = target_counts[s][c]
                current = allocated_counts[s][c]
                deficit = target - current
                if target > 0:
                    deficit_score += (deficit / target) * cnt
                else:
                    deficit_score += deficit * cnt

            if deficit_score > best_score:
                best_score = deficit_score
                best_split = s

        assert best_split is not None
        group_split_assignment[gid] = best_split
        for c, cnt in grp_counts.items():
            allocated_counts[best_split][c] += cnt

    # 5. Build Split Example Lists
    train_groups = {g for g, s in group_split_assignment.items() if s == "train"}
    val_groups = {g for g, s in group_split_assignment.items() if s == "val"}
    test_groups = {g for g, s in group_split_assignment.items() if s == "test"}

    # Strict Group Isolation Check
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("Group isolation violation: a source_group_id appeared in multiple splits.")

    train_examples = [ex for g in sorted(train_groups) for ex in group_to_examples[g]]
    val_examples = [ex for g in sorted(val_groups) for ex in group_to_examples[g]]
    test_examples = [ex for g in sorted(test_groups) for ex in group_to_examples[g]]

    # 6. Validate Split Sizes and Class Coverage
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

    train_classes = set(ex.canonical_intent for ex in train_examples)
    test_classes = set(ex.canonical_intent for ex in test_examples) if test_examples else set()

    # Verify every overall class is present in train split
    missing_train_classes = set(distinct_classes) - train_classes
    if missing_train_classes:
        raise ValueError(
            f"Stratified group splitting failed: class(es) {sorted(missing_train_classes)} missing from train split. "
            "Dataset has insufficient groups per class to achieve stratified split."
        )

    # Verify test split has sufficient class coverage
    if require_multi_class and len(test_classes) < 2:
        raise ValueError(
            f"Invalid single-class test split produced (test classes: {sorted(test_classes)}). "
            "Multi-class intent evaluation requires at least 2 distinct classes in test split."
        )

    # 7. Pre-training Leakage Verification
    check_split_leakage(train_examples, val_examples, test_examples)

    return train_examples, val_examples, test_examples
