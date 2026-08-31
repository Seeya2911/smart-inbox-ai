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
    for index, text in enumerate(normalized):
        tokens = set(text.split())
        token_sets.append(tokens)
        for token in tokens:
            token_to_indices[token].add(index)

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
        group_to_examples[f"cluster-{component_ids[index]}"].append(example)

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


# ---------------------------------------------------------------------------
# Multi-output email dataset splitter (CanonicalEmailExample)
# ---------------------------------------------------------------------------

from ml.data_quality import check_split_leakage_email  # noqa: E402
from ml.schema import CanonicalEmailExample  # noqa: E402

_JOINT_RARE_THRESHOLD = 3  # combinations with fewer examples fall back to intent-only


def _near_dup_group_ids_email(
    examples: List[CanonicalEmailExample],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> Dict[int, int]:
    """Near-duplicate component IDs for CanonicalEmailExample.

    Identical logic to _near_duplicate_group_ids but uses ex.full_text
    and ex.source_group_id / ex.id for source-group isolation.
    """
    if not examples:
        return {}

    parent = list(range(len(examples)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Source-group isolation
    source_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        source_groups[ex.source_group_id or ex.id].append(idx)
    for indices in source_groups.values():
        first = indices[0]
        for idx in indices[1:]:
            union(first, idx)

    normalized = [normalize_text(ex.full_text) for ex in examples]

    # Exact normalized duplicate clustering
    first_by_norm: Dict[str, int] = {}
    for idx, norm in enumerate(normalized):
        prev = first_by_norm.get(norm)
        if prev is None:
            first_by_norm[norm] = idx
        else:
            union(prev, idx)

    # Token index for near-duplicate search (skip ultra-frequent stopwords)
    token_to_indices: Dict[str, Set[int]] = defaultdict(set)
    token_sets: List[Set[str]] = []
    for idx, norm in enumerate(normalized):
        tokens = set(norm.split())
        token_sets.append(tokens)
        for tok in tokens:
            token_to_indices[tok].add(idx)

    max_freq = max(20, int(len(examples) * 0.20))
    informative_tokens = {tok: idxs for tok, idxs in token_to_indices.items() if 2 <= len(idxs) <= max_freq and len(tok) >= 3}

    for idx, norm in enumerate(normalized):
        if len(norm) < 15:
            continue
        candidates: Set[int] = set()
        for tok in token_sets[idx]:
            if tok in informative_tokens:
                candidates.update(informative_tokens[tok])
        candidates.discard(idx)

        len1 = len(norm)
        for other in sorted(candidates):
            if other <= idx:
                continue
            len2 = len(normalized[other])
            if len2 < 15:
                continue
            if min(len1, len2) / max(len1, len2) < 0.70:
                continue
            word_sim = _word_jaccard(norm, normalized[other])
            if word_sim >= WORD_JACCARD_THRESHOLD:
                union(idx, other)
            elif word_sim >= 0.40:
                char_sim = compute_ngram_jaccard(norm, normalized[other], n=4)
                if char_sim >= threshold:
                    union(idx, other)

    root_to_comp: Dict[int, int] = {}
    comp_ids: Dict[int, int] = {}
    next_comp = 0
    for idx in range(len(examples)):
        root = find(idx)
        if root not in root_to_comp:
            root_to_comp[root] = next_comp
            next_comp += 1
        comp_ids[idx] = root_to_comp[root]
    return comp_ids


def _print_split_distributions(
    train: List[CanonicalEmailExample],
    val: List[CanonicalEmailExample],
    test: List[CanonicalEmailExample],
    all_intents: List[str],
    all_priorities: List[str],
    joint_strat: bool,
    rare_combos: List[str],
) -> None:
    """Print intent, priority, and intent×priority distributions for each split."""
    sep = "=" * 62

    def dist(exs: List[CanonicalEmailExample], attr: str) -> Dict[str, int]:
        from collections import Counter
        return dict(sorted(Counter(getattr(e, attr) for e in exs).items()))

    print(f"\n{sep}")
    print("  SPLIT DISTRIBUTIONS")
    strat_note = "joint intent×priority" if joint_strat else "intent-only (fallback)"
    print(f"  Stratification: {strat_note}")
    if rare_combos:
        print(f"  Rare combos excluded from joint strat: {rare_combos[:10]}")
    print(sep)

    for name, exs in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n  [{name}]  n={len(exs)}")
        print("   Intent distribution:")
        idist = dist(exs, "intent")
        for cls in sorted(all_intents):
            cnt = idist.get(cls, 0)
            print(f"     {cls:<20} {cnt:>4}")
        print("   Priority distribution:")
        pdist = dist(exs, "priority")
        for p in sorted(all_priorities):
            cnt = pdist.get(p, 0)
            print(f"     {p:<20} {cnt:>4}")

    print(f"\n  INTENT × PRIORITY DISTRIBUTION BY SPLIT")
    priorities = sorted(all_priorities)
    for name, exs in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n  [{name}]  n={len(exs)}")
        header = f"    {'intent':<22}" + "".join(f"{p:>8}" for p in priorities)
        print(header)
        print("    " + "-" * (22 + 8 * len(priorities)))
        from collections import Counter
        joint_cnt: Dict[str, int] = Counter(
            f"{e.intent}|{e.priority}" for e in exs
        )
        for intent in sorted(all_intents):
            row = f"    {intent:<22}" + "".join(
                f"{joint_cnt.get(f'{intent}|{p}', 0):>8}" for p in priorities
            )
            print(row)

    print()


def split_email_dataset(
    examples: List[CanonicalEmailExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    joint_rare_threshold: int = _JOINT_RARE_THRESHOLD,
    print_distributions: bool = True,
) -> Tuple[
    List[CanonicalEmailExample],
    List[CanonicalEmailExample],
    List[CanonicalEmailExample],
]:
    """Deterministically split a multi-output email dataset into train/val/test.

    Stratification strategy
    -----------------------
    1. **Primary**: stratify on the joint ``intent|priority`` key, which
       preserves both output distributions simultaneously.
    2. **Fallback**: if any intent×priority combination has fewer than
       ``joint_rare_threshold`` examples, switch to intent-only stratification
       and report which combinations were too rare.

    Near-duplicate and source-group isolation is applied before any bin-packing,
    so related examples are always assigned to the same split.

    After splitting, intent distribution, priority distribution, and the full
    intent×priority matrix are printed for each split (when print_distributions
    is True).

    Raises
    ------
    ValueError
        If ratios do not sum to 1.0, the dataset is empty, or any split is empty.
    DataLeakageError
        If any ID / text / near-duplicate overlap is found between splits.
    """
    from ml.schema import ALLOWED_INTENTS, ALLOWED_PRIORITIES

    if not examples:
        raise ValueError("Cannot split an empty dataset.")
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0; got {train_ratio}+{val_ratio}+{test_ratio}={total_ratio}"
        )

    all_intents = sorted(ALLOWED_INTENTS)
    all_priorities = sorted(ALLOWED_PRIORITIES)

    # ---- Determine stratification key ----
    from collections import Counter
    joint_counts: Counter[str] = Counter(f"{e.intent}|{e.priority}" for e in examples)
    rare_combos = [k for k, v in joint_counts.items() if v < joint_rare_threshold]
    use_joint = len(rare_combos) == 0

    strat_key_fn = (
        (lambda ex: f"{ex.intent}|{ex.priority}") if use_joint
        else (lambda ex: ex.intent)
    )

    # Overall class counts (for bin-packing targets)
    overall_class_counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        overall_class_counts[strat_key_fn(ex)] += 1

    distinct_classes = sorted(overall_class_counts)

    # ---- Near-duplicate grouping ----
    comp_ids = _near_dup_group_ids_email(examples, threshold=near_duplicate_threshold)
    group_to_examples: Dict[str, List[CanonicalEmailExample]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        group_to_examples[f"cluster-{comp_ids[idx]}"].append(ex)

    group_ids = sorted(group_to_examples)
    group_class_counts: Dict[str, Dict[str, int]] = {}
    for gid, gexs in group_to_examples.items():
        counts: Dict[str, int] = defaultdict(int)
        for ex in gexs:
            counts[strat_key_fn(ex)] += 1
        group_class_counts[gid] = counts

    # ---- Stratified bin-packing ----
    split_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    active_splits = [s for s, r in split_ratios.items() if r > 0]
    target_counts = {
        split: {cls: overall_class_counts[cls] * split_ratios[split] for cls in distinct_classes}
        for split in active_splits
    }
    allocated_counts = {
        split: {cls: 0.0 for cls in distinct_classes} for split in active_splits
    }

    rng = np.random.RandomState(seed)

    # Sort rarest classes first (ensures minority classes are placed carefully)
    def group_rarity_key(gid: str) -> int:
        return min(overall_class_counts[cls] for cls in group_class_counts[gid])

    tier_groups: Dict[int, List[str]] = defaultdict(list)
    for gid in group_ids:
        tier_groups[group_rarity_key(gid)].append(gid)

    ordered_groups: List[str] = []
    for tier in sorted(tier_groups):
        candidates = tier_groups[tier]
        rng.shuffle(candidates)
        ordered_groups.extend(candidates)

    group_split: Dict[str, str] = {}
    for gid in ordered_groups:
        gcounts = group_class_counts[gid]
        best_split: Optional[str] = None
        best_score = -float("inf")
        for split in active_splits:
            deficit_score = 0.0
            for cls, cnt in gcounts.items():
                target = target_counts[split].get(cls, 0)
                current = allocated_counts[split].get(cls, 0.0)
                deficit = target - current
                deficit_score += (deficit / target) * cnt if target > 0 else deficit * cnt
            if deficit_score > best_score:
                best_score = deficit_score
                best_split = split
        assert best_split is not None
        group_split[gid] = best_split
        for cls, cnt in gcounts.items():
            allocated_counts[best_split][cls] = allocated_counts[best_split].get(cls, 0.0) + cnt

    train_groups = {g for g, s in group_split.items() if s == "train"}
    val_groups = {g for g, s in group_split.items() if s == "val"}
    test_groups = {g for g, s in group_split.items() if s == "test"}

    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("Group isolation violation: a group appeared in multiple splits.")

    train_ex = [ex for g in sorted(train_groups) for ex in group_to_examples[g]]
    val_ex = [ex for g in sorted(val_groups) for ex in group_to_examples[g]]
    test_ex = [ex for g in sorted(test_groups) for ex in group_to_examples[g]]

    if not train_ex:
        raise ValueError(
            f"Stratified split produced empty train set (val={len(val_ex)}, test={len(test_ex)}). "
            "Dataset has too few groups per class."
        )
    if val_ratio > 0 and not val_ex:
        raise ValueError(
            f"Stratified split produced empty val set (train={len(train_ex)}, test={len(test_ex)})."
        )
    if test_ratio > 0 and not test_ex:
        raise ValueError(
            f"Stratified split produced empty test set (train={len(train_ex)}, val={len(val_ex)})."
        )

    # ---- Verify all classes in train ----
    if use_joint:
        train_keys = {f"{ex.intent}|{ex.priority}" for ex in train_ex}
        missing = set(distinct_classes) - train_keys
    else:
        train_keys = {ex.intent for ex in train_ex}
        missing = set(all_intents) - train_keys

    if missing:
        raise ValueError(
            f"Stratified split failed: class(es) missing from train split: {sorted(missing)}. "
            "Too few examples per class to achieve stratified split."
        )

    # ---- Leakage check ----
    check_split_leakage_email(train_ex, val_ex, test_ex)

    # ---- Print distributions ----
    if print_distributions:
        _print_split_distributions(
            train_ex, val_ex, test_ex,
            all_intents=all_intents,
            all_priorities=all_priorities,
            joint_strat=use_joint,
            rare_combos=rare_combos,
        )

    return train_ex, val_ex, test_ex

