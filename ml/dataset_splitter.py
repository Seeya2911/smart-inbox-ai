"""Deterministic Train/Validation/Test Dataset Splitting Module.

Provides deterministic splitting with fixed random seeds, stratified intent distribution,
and group-level grouping (keeping related or augmented/translated examples together).
Automatically verifies split leakage safety prior to returning.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from ml.data_quality import check_split_leakage
from ml.schema import CanonicalIntentExample


def split_intent_dataset(
    examples: List[CanonicalIntentExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[CanonicalIntentExample], List[CanonicalIntentExample], List[CanonicalIntentExample]]:
    """Split dataset deterministically into train, validation, and test splits.

    Args:
        examples: Clean list of CanonicalIntentExample instances.
        train_ratio: Proportion for training split (default 0.70).
        val_ratio: Proportion for validation split (default 0.15).
        test_ratio: Proportion for test split (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        (train_examples, val_examples, test_examples)
    """
    if not examples:
        raise ValueError("Cannot split an empty dataset.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}")

    # Check if grouping is present
    has_groups = any(bool(ex.source_group_id) for ex in examples)

    if has_groups:
        # Group examples by source_group_id (falling back to example ID if source_group_id is empty)
        group_to_examples: Dict[str, List[CanonicalIntentExample]] = defaultdict(list)
        for ex in examples:
            gid = ex.source_group_id if ex.source_group_id else ex.source_example_id
            group_to_examples[gid].append(ex)

        groups = sorted(group_to_examples.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(groups)

        num_groups = len(groups)
        num_train_groups = int(round(num_groups * train_ratio))
        num_val_groups = int(round(num_groups * val_ratio))

        train_groups = groups[:num_train_groups]
        val_groups = groups[num_train_groups : num_train_groups + num_val_groups]
        test_groups = groups[num_train_groups + num_val_groups :]

        train_examples = [ex for g in train_groups for ex in group_to_examples[g]]
        val_examples = [ex for g in val_groups for ex in group_to_examples[g]]
        test_examples = [ex for g in test_groups for ex in group_to_examples[g]]
    else:
        # Stratified random split
        indices = np.arange(len(examples))
        labels = [ex.canonical_intent for ex in examples]

        # Check if stratification is possible (each class needs >= 2 instances)
        label_counts = defaultdict(int)
        for lbl in labels:
            label_counts[lbl] += 1
        can_stratify = all(count >= 2 for count in label_counts.values()) and len(label_counts) > 1

        val_test_ratio = val_ratio + test_ratio
        val_relative_ratio = val_ratio / val_test_ratio if val_test_ratio > 0 else 0.5

        train_idx, val_test_idx = train_test_split(
            indices,
            test_size=val_test_ratio,
            random_state=seed,
            stratify=labels if can_stratify else None,
        )

        val_test_labels = [labels[i] for i in val_test_idx]
        val_test_counts = defaultdict(int)
        for lbl in val_test_labels:
            val_test_counts[lbl] += 1
        can_stratify_val_test = all(c >= 2 for c in val_test_counts.values()) and len(val_test_counts) > 1

        val_sub_idx, test_sub_idx = train_test_split(
            val_test_idx,
            test_size=(1.0 - val_relative_ratio),
            random_state=seed,
            stratify=val_test_labels if can_stratify_val_test else None,
        )

        train_examples = [examples[i] for i in train_idx]
        val_examples = [examples[i] for i in val_sub_idx]
        test_examples = [examples[i] for i in test_sub_idx]

    # Verify split safety (raises DataLeakageError if leakage exists)
    check_split_leakage(train_examples, val_examples, test_examples)

    return train_examples, val_examples, test_examples
