"""Prepare deduplicated raw email batches for external Manus labeling.

The exporter deliberately emits only the three fields used by the external
labeling workflow: subject, body, and source. Smart Inbox intent/priority are
NOT assigned here. A deterministic manifest preserves the source IDs so the
labeled workbook can be joined back to the canonical corpus after Manus
returns it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook


SIGNATURE_RE = re.compile(
    r"\n\s*(?:thanks|best regards|kind regards|cheers|sincerely|warm regards|regards|yours truly|best),?\s*\n.*$",
    re.IGNORECASE | re.DOTALL,
)
DISCLAIMER_RES = [
    re.compile(r"this email and any files transmitted with it are confidential.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"if you have received this email in error please notify.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"the contents of this email message and any attachments are intended solely.*", re.IGNORECASE | re.DOTALL),
]


def normalize_for_dedup(subject: str, body: str) -> str:
    """Normalize content conservatively for exact duplicate detection."""
    text = f"{subject}\n{body}".strip()
    lines = [line for line in text.splitlines() if not line.strip().startswith(">")]
    text = "\n".join(lines)
    for pattern in DISCLAIMER_RES:
        text = pattern.sub("", text)
    text = SIGNATURE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def iter_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_input_file"] = str(path)
                record["_input_line"] = line_number
                yield record


def prepare_records(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {"rows_read": 0, "empty_removed": 0, "exact_duplicates_removed": 0}

    for record in iter_records(paths):
        stats["rows_read"] += 1
        subject = str(record.get("subject", "")).strip()
        body = str(record.get("body", record.get("text", ""))).strip()
        source = str(record.get("source", "")).strip()
        if not subject and not body:
            stats["empty_removed"] += 1
            continue
        if not source:
            raise ValueError("Every email must have a non-empty source")

        normalized = normalize_for_dedup(subject, body)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            stats["exact_duplicates_removed"] += 1
            continue
        seen.add(digest)
        output.append(
            {
                "id": str(record.get("id", record.get("source_example_id", ""))).strip(),
                "subject": subject,
                "body": body,
                "source": source,
            }
        )

    return output, stats


def write_batches(records: list[dict[str, Any]], output_dir: Path, batch_size: int) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_number = start // batch_size + 1
        batch = records[start : start + batch_size]
        path = output_dir / f"manus_batch_{batch_number:04d}.xlsx"

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "emails"
        sheet.append(["subject", "body", "source"])
        for record in batch:
            sheet.append([record["subject"], record["body"], record["source"]])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.column_dimensions["A"].width = 42
        sheet.column_dimensions["B"].width = 110
        sheet.column_dimensions["C"].width = 22
        workbook.save(path)

        for offset, record in enumerate(batch, start=2):
            manifest.append(
                {
                    "batch": path.name,
                    "excel_row": offset,
                    "id": record["id"],
                    "source": record["source"],
                }
            )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deduplicated email XLSX batches for Manus labeling.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Raw canonical JSONL files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    records, stats = prepare_records(args.inputs)
    manifest = write_batches(records, args.output_dir, args.batch_size)
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.update({"rows_after_dedup": len(records), "batches": len({row['batch'] for row in manifest})})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
