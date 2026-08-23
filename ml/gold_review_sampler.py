"""Create a deterministic human-review queue from weak-labeled email data.

The sampler deliberately does not create gold labels. It selects examples for
human annotation while preserving weak labels and provenance as context. The
sample is stratified across source, weak intent/priority, and weak-signal
population so the gold set is not dominated by easy rule matches.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield row


def _population(rule_score: float, high_threshold: float, low_threshold: float) -> str:
    if rule_score >= high_threshold:
        return "high_score"
    if rule_score >= low_threshold:
        return "ambiguous"
    return "low_signal"


def build_review_queue(
    rows: list[dict[str, Any]],
    count: int = 400,
    seed: int = 42,
    high_threshold: float = 3.0,
    low_threshold: float = 1.0,
) -> list[dict[str, Any]]:
    """Select a deterministic, diversity-oriented human review sample."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not rows:
        return []

    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not str(row.get("body", row.get("text", ""))).strip():
            continue
        try:
            score = float(row.get("rule_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        candidate = dict(row)
        candidate["review_population"] = _population(score, high_threshold, low_threshold)
        candidate["review_intent"] = ""
        candidate["review_priority"] = ""
        candidate["review_notes"] = ""
        candidate["reviewer"] = ""
        candidates.append(candidate)

    if not candidates:
        return []

    # Stratify on the dimensions that matter for detecting labeling bias.
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            str(row.get("source", "unknown")),
            str(row.get("intent", "other")),
            str(row.get("priority", "low")),
            str(row["review_population"]),
        )
        buckets[key].append(row)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Round-robin over sorted strata prevents the largest source/population
    # from consuming the entire review budget.
    ordered_keys = sorted(buckets)
    selected: list[dict[str, Any]] = []
    while len(selected) < min(count, len(candidates)):
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) >= min(count, len(candidates)):
                    break
        if not progressed:
            break

    # Shuffle only the final presentation order; selection remains deterministic.
    rng.shuffle(selected)
    return selected


def write_review_queue(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic human-review queue.")
    parser.add_argument("--input", type=Path, required=True, help="Weak-labeled JSONL input")
    parser.add_argument("--output", type=Path, required=True, help="Review queue JSONL output")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high-threshold", type=float, default=3.0)
    parser.add_argument("--low-threshold", type=float, default=1.0)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input))
    selected = build_review_queue(
        rows,
        count=args.count,
        seed=args.seed,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
    )
    write_review_queue(selected, args.output)
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "review_rows": len(selected),
                "output": str(args.output),
                "seed": args.seed,
                "gold_labels_created": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
