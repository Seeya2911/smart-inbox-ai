"""Analyze unlabeled email corpora before weak labeling.

The coverage report is intentionally label-agnostic: source-native labels are
reported as metadata only and are never mapped to Smart Inbox intent/priority.
The goal is to expose source imbalance, missing/empty content, language mix,
length characteristics, and exact cross-source content overlap before labels
are created.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalize_content(subject: str, body: str) -> str:
    """Return deterministic normalized subject/body content for overlap checks."""
    text = f"{subject}\n{body}".strip()
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from one or more JSONL files."""
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object in {path}:{line_number}")
                yield row


def analyze_corpora(paths: Iterable[Path]) -> dict[str, Any]:
    """Build a deterministic coverage report for raw/unlabeled corpus JSONL."""
    source_counts: Counter[str] = Counter()
    source_labels: dict[str, Counter[str]] = defaultdict(Counter)
    languages: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    duplicate_sources: dict[str, set[str]] = defaultdict(set)
    source_row_counts: Counter[str] = Counter()
    empty_rows = 0
    body_lengths: list[int] = []
    subject_lengths: list[int] = []
    total = 0

    for row in iter_jsonl(paths):
        total += 1
        source = str(row.get("source", row.get("source_dataset", "unknown"))).strip() or "unknown"
        source_counts[source] += 1
        source_row_counts[source] += 1

        language = str(row.get("language", "unknown")).strip().lower() or "unknown"
        languages[language] += 1

        native_label = row.get("source_label", row.get("label", ""))
        if native_label is not None and str(native_label).strip():
            source_labels[source][str(native_label).strip().lower()] += 1

        subject = str(row.get("subject", "") or "")
        body = str(row.get("body", row.get("text", row.get("email", ""))) or "")
        if not (subject.strip() or body.strip()):
            empty_rows += 1
        subject_lengths.append(len(subject))
        body_lengths.append(len(body))

        normalized = normalize_content(subject, body)
        if normalized:
            duplicate_counts[normalized] += 1
            duplicate_sources[normalized].add(source)

    exact_duplicate_groups = sum(1 for count in duplicate_counts.values() if count > 1)
    cross_source_duplicate_groups = sum(
        1 for normalized, count in duplicate_counts.items()
        if count > 1 and len(duplicate_sources[normalized]) > 1
    )

    def stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        ordered = sorted(values)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        return {
            "min": ordered[0],
            "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 2),
            "median": median,
        }

    return {
        "total_rows": total,
        "source_counts": dict(sorted(source_counts.items())),
        "source_native_labels": {
            source: dict(sorted(labels.items())) for source, labels in sorted(source_labels.items())
        },
        "language_counts": dict(sorted(languages.items())),
        "empty_content_rows": empty_rows,
        "subject_length": stats(subject_lengths),
        "body_length": stats(body_lengths),
        "unique_normalized_content": len(duplicate_counts),
        "exact_duplicate_groups": exact_duplicate_groups,
        "cross_source_exact_duplicate_groups": cross_source_duplicate_groups,
        "source_row_counts": dict(sorted(source_row_counts.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze unlabeled email corpus coverage.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL corpus files")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze_corpora(args.inputs)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
