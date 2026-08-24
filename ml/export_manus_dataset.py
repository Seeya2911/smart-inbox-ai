"""Build a deterministic Manus-ready XLSX from the canonical raw email pool.

The exporter intentionally contains NO Smart Inbox labels. Manus is the teacher:
this file only selects clean, deduplicated email text and preserves provenance.

Input JSONL records are expected to contain at least ``subject``, ``body``, and
``source``. Optional ``id`` and ``source_example_id`` fields are retained in the
manifest, not exposed to the labeler. The output workbook has exactly three
labeling columns: subject, body, source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook


def _norm(text: str) -> str:
    text = text.lower().replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _content_key(row: dict[str, Any]) -> str:
    subject = _norm(str(row.get("subject", "")))
    body = _norm(str(row.get("body", "")))
    return hashlib.sha256(f"{subject}\n{body}".encode("utf-8")).hexdigest()


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {"input_rows": 0, "empty_removed": 0, "exact_duplicates_removed": 0}
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["input_rows"] += 1
                row = json.loads(line)
                subject = str(row.get("subject", "")).strip()
                body = str(row.get("body", "")).strip()
                source = str(row.get("source", "")).strip()
                if not subject and not body:
                    stats["empty_removed"] += 1
                    continue
                if not source:
                    raise ValueError("Every row must have a non-empty source")
                row["subject"] = subject
                row["body"] = body
                row["source"] = source
                key = _content_key(row)
                if key in seen:
                    stats["exact_duplicates_removed"] += 1
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda r: (str(r["source"]), _norm(str(r["subject"])), str(r.get("id", ""))))
    return rows, stats


def export_xlsx(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "emails"
    sheet.append(["subject", "body", "source"])
    for row in rows:
        sheet.append([row["subject"], row["body"], row["source"]])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.column_dimensions["A"].width = 45
    sheet.column_dimensions["B"].width = 100
    sheet.column_dimensions["C"].width = 20
    workbook.save(output)


def write_manifest(rows: list[dict[str, Any]], stats: dict[str, int], output: Path) -> None:
    source_counts: dict[str, int] = {}
    for row in rows:
        source = row["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    payload = {**stats, "output_rows": len(rows), "source_counts": dict(sorted(source_counts.items()))}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export clean canonical emails for Manus labeling")
    parser.add_argument("inputs", nargs="+", type=Path, help="canonical JSONL files")
    parser.add_argument("--output", type=Path, required=True, help="Manus-ready XLSX")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    rows, stats = load_rows(args.inputs)
    if not rows:
        raise SystemExit("No usable email rows found")
    export_xlsx(rows, args.output)
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    write_manifest(rows, stats, manifest)
    print(json.dumps({**stats, "output_rows": len(rows), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
