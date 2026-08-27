"""Deterministic group-aware splitting for the production intent + priority dataset.

The splitter keeps source groups, exact duplicates, and strong near-duplicates in
one split. Stratification is performed over the joint ``intent|priority`` label so
both prediction heads retain useful coverage. The split is independent of the LLM
and therefore cannot change labels.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ml.data_quality import compute_ngram_jaccard, normalize_text
from ml.schema import CanonicalEmailExample

CHAR_NGRAM_THRESHOLD = 0.85
TOKEN_JACCARD_THRESHOLD = 0.80


def _token_jaccard(text1: str, text2: str) -> float:
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def _group_ids(examples: List[CanonicalEmailExample]) -> Dict[int, int]:
    if not examples:
        return {}

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

    source_groups: Dict[str, List[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        source_groups[example.source_group_id or example.source_example_id or example.id].append(index)
    for indices in source_groups.values():
        first = indices[0]
        for index in indices[1:]:
            union(first, index)

    normalized = [normalize_text(example.full_text) for example in examples]
    first_by_text: Dict[str, int] = {}
    for index, text in enumerate(normalized):
        previous = first_by_text.get(text)
        if previous is None:
            first_by_text[text] = index
        else:
            union(previous, index)

    token_sets = [set(text.split()) for text in normalized]
    token_index: Dict[str, Set[int]] = defaultdict(set)
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            token_index[token].add(index)

    for index, text in enumerate(normalized):
        if len(text) < 15:
            continue
        candidates: Set[int] = set()
        for token in token_sets[index]:
            candidates.update(token_index[token])
        for other in sorted(candidates):
            if other <= index or len(normalized[other]) < 15:
                continue
            len1, len2 = len(text), len(normalized[other])
            if min(len1, len2) / max(len1, len2) < 0.70:
                continue
            if (
                _token_jaccard(text, normalized[other]) >= TOKEN_JACCARD_THRESHOLD
                or compute_ngram_jaccard(text, normalized[other], n=4) >= CHAR_NGRAM_THRESHOLD
            ):
                union(index, other)

    root_to_component: Dict[int, int] = {}
    result: Dict[int, int] = {}
    next_component = 0
    for index in range(len(examples)):
        root = find(index)
        if root not in root_to_component:
            root_to_component[root] = next_component
            next_component += 1
        result[index] = root_to_component[root]
    return result


def split_multi_output_dataset(
    examples: List[CanonicalEmailExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[CanonicalEmailExample], List[CanonicalEmailExample], List[CanonicalEmailExample]]:
    """Split a canonical production corpus without leakage."""
    if not examples:
        raise ValueError("Cannot split an empty dataset")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    label_counts: Dict[str, int] = defaultdict(int)
    for example in examples:
        label_counts[f"{example.intent}|{example.priority}"] += 1
    if len({example.intent for example in examples}) < 2:
        raise ValueError("Production intent dataset must contain at least two intent classes")
    if len({example.priority for example in examples}) < 2:
        raise ValueError("Production priority dataset must contain at least two priority classes")

    components = _group_ids(examples)
    groups: Dict[str, List[CanonicalEmailExample]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[f"cluster-{components[index]}"] .append(example)

    group_labels: Dict[str, Dict[str, int]] = {}
    for group_id, group_examples in groups.items():
        counts: Dict[str, int] = defaultdict(int)
        for example in group_examples:
            counts[f"{example.intent}|{example.priority}"] += 1
        group_labels[group_id] = counts

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    active = [name for name, ratio in ratios.items() if ratio > 0]
    targets = {
        split: {label: label_counts[label] * ratios[split] for label in label_counts}
        for split in active
    }
    allocated = {split: defaultdict(float) for split in active}

    rng = np.random.RandomState(seed)
    ordered = sorted(groups, key=lambda group: min(label_counts[label] for label in group_labels[group]))
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: min(label_counts[label] for label in group_labels[group]))

    assignment: Dict[str, str] = {}
    for group_id in ordered:
        best_split: Optional[str] = None
        best_score = -float("inf")
        for split in active:
            score = 0.0
            for label, count in group_labels[group_id].items():
                target = targets[split][label]
                deficit = target - allocated[split][label]
                score += (deficit / target) * count if target else deficit * count
            if score > best_score:
                best_score = score
                best_split = split
        assert best_split is not None
        assignment[group_id] = best_split
        for label, count in group_labels[group_id].items():
            allocated[best_split][label] += count

    result: Dict[str, List[CanonicalEmailExample]] = {split: [] for split in active}
    for group_id in sorted(groups):
        result[assignment[group_id]].extend(groups[group_id])

    train = result.get("train", [])
    val = result.get("val", [])
    test = result.get("test", [])
    if not train or (val_ratio > 0 and not val) or (test_ratio > 0 and not test):
        raise ValueError("Leakage-aware splitting could not produce all requested non-empty splits")
    return train, val, test


def load_jsonl(path: Path) -> List[CanonicalEmailExample]:
    with path.open("r", encoding="utf-8") as handle:
        return [CanonicalEmailExample.from_dict(json.loads(line)) for line in handle if line.strip()]


def write_jsonl(path: Path, examples: List[CanonicalEmailExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    return len(examples)


def split_file(input_path: Path, output_dir: Path, seed: int = 42) -> Dict[str, int]:
    examples = load_jsonl(input_path)
    train, val, test = split_multi_output_dataset(examples, seed=seed)
    counts = {
        "train": write_jsonl(output_dir / "train.jsonl", train),
        "val": write_jsonl(output_dir / "val.jsonl", val),
        "test": write_jsonl(output_dir / "test.jsonl", test),
    }
    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Leakage-aware production multi-output splitter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(split_file(args.input, args.output_dir, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
