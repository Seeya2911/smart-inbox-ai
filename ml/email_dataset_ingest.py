"""Download and canonicalize public email corpora into raw JSONL.

This stage intentionally does NOT assign intent or priority labels. It preserves
source provenance so the later weak-labeling pipeline can label the same emails
without pretending source datasets contain our production taxonomy.

The downloader uses the Hugging Face datasets-server rows API, avoiding a hard
runtime dependency on the ``datasets`` package. It is resumable by offset and
supports bounded downloads for development as well as full-corpus ingestion.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class RawEmailRecord:
    """Unlabeled canonical representation of one public email."""

    id: str
    subject: str
    body: str
    source: str
    source_example_id: str
    source_split: str
    is_synthetic: bool = False
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _request_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        f"{HF_ROWS_API}?{params}",
        headers={"User-Agent": "smart-inbox-ai-dataset-ingest/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    rows = payload.get("rows", [])
    return [row.get("row", {}) for row in rows]


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_enron(row: dict[str, Any], source_example_id: str, split: str) -> RawEmailRecord:
    # HF mirrors differ slightly in field names, so accept common representations.
    text = _first_text(row, "text", "body", "email")
    subject = _first_text(row, "subject")
    if not subject and text:
        for line in text.splitlines():
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
                break
    return RawEmailRecord(
        id=f"enron-{source_example_id}",
        subject=subject,
        body=text,
        source="enron",
        source_example_id=source_example_id,
        source_split=split,
    )


def normalize_spamassassin(row: dict[str, Any], source_example_id: str, split: str) -> RawEmailRecord:
    text = _first_text(row, "text", "body", "email")
    subject = _first_text(row, "subject")
    if not subject and text:
        for line in text.splitlines():
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
                break
    return RawEmailRecord(
        id=f"spamassassin-{source_example_id}",
        subject=subject,
        body=text,
        source="spam_corpus",
        source_example_id=source_example_id,
        source_split=split,
    )


def iter_source(
    *, dataset: str, config: str, split: str, source: str, max_rows: int | None, batch_size: int, sleep_seconds: float
) -> Iterable[RawEmailRecord]:
    offset = 0
    seen = 0
    normalizer = normalize_enron if source == "enron" else normalize_spamassassin

    while max_rows is None or seen < max_rows:
        length = batch_size if max_rows is None else min(batch_size, max_rows - seen)
        rows = _request_rows(dataset, config, split, offset, length)
        if not rows:
            break
        for index, row in enumerate(rows):
            source_id = str(row.get("id", offset + index))
            yield normalizer(row, source_id, split)
            seen += 1
            if max_rows is not None and seen >= max_rows:
                break
        offset += len(rows)
        if len(rows) < length:
            break
        if sleep_seconds:
            time.sleep(sleep_seconds)


def write_jsonl(records: Iterable[RawEmailRecord], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            if not (record.subject.strip() or record.body.strip()):
                continue
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download public email corpora into unlabeled canonical JSONL.")
    parser.add_argument("--source", choices=["enron", "spamassassin"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None, help="Bound the download; omit for the complete available split.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be positive when supplied")

    if args.source == "enron":
        dataset, config, split = "corbt/enron-emails", "default", "train"
    else:
        dataset, config, split = "talby/spamassassin", "text", "train"

    records = iter_source(
        dataset=dataset,
        config=config,
        split=split,
        source=args.source,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )
    count = write_jsonl(records, args.output)
    print(json.dumps({"output": str(args.output), "source": args.source, "rows_written": count}, indent=2))


if __name__ == "__main__":
    main()
