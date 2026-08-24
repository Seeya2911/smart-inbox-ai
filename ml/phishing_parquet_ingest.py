"""Ingest retained phishing/spam source emails from the verified Parquet artifact.

This module deliberately does not assign Smart Inbox intent or priority. The
native binary label is retained only as source provenance. We use the verified
Parquet artifact exposed by the public Hugging Face dataset instead of guessing
individual CSV URLs, which avoids the broken-file/404 path that affected the
previous downloader.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DATASET_URL = (
    "https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets/"
    "resolve/main/train.parquet?download=true"
)
SOURCE_ALLOWLIST = ("TREC-05", "TREC-06", "TREC-07", "CEAS-08", "Ling")


def _delay(error: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return min(60.0, max(0.5, float(retry_after)))
        except ValueError:
            pass
    return min(60.0, 2.0**attempt)


def download_dataset(destination: Path, max_retries: int = 5) -> None:
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "smart-inbox-ai-phishing-ingest/1.0"},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # nosec B310 - fixed HTTPS URL
                with destination.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            return
        except urllib.error.HTTPError as error:
            if error.code not in {429, 502, 503, 504} or attempt >= max_retries:
                raise
            time.sleep(_delay(error, attempt))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_records(parquet_path: Path, max_rows: int) -> list[dict[str, Any]]:
    base, remainder = divmod(max_rows, len(SOURCE_ALLOWLIST))
    quotas = {
        source: base + (index < remainder)
        for index, source in enumerate(SOURCE_ALLOWLIST)
    }
    counts = {source: 0 for source in SOURCE_ALLOWLIST}
    records: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(parquet_path)
    columns = ["text", "subject", "label", "dataset_name"]
    global_index = 0

    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group, columns=columns)
        for row in table.to_pylist():
            source_dataset = _text(row.get("dataset_name"))
            if source_dataset not in SOURCE_ALLOWLIST or counts[source_dataset] >= quotas[source_dataset]:
                global_index += 1
                continue

            subject = _text(row.get("subject"))
            body = _text(row.get("text"))
            if not subject and not body:
                global_index += 1
                continue

            source_example_id = str(global_index)
            records.append(
                {
                    "id": f"phishing_corpus_{source_dataset}:{source_example_id}",
                    "subject": subject,
                    "body": body,
                    "source": "phishing_corpus",
                    "source_example_id": f"{source_dataset}:{source_example_id}",
                    "source_split": "train",
                    "source_dataset": source_dataset,
                    "source_label": _text(row.get("label")),
                    "is_synthetic": False,
                    "language": "en",
                }
            )
            counts[source_dataset] += 1
            global_index += 1
            if len(records) >= max_rows:
                return records

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retained phishing corpus from verified Parquet data")
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_rows < 1:
        raise SystemExit("--max-rows must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smart_inbox_phishing_") as temp_dir:
        parquet_path = Path(temp_dir) / "train.parquet"
        download_dataset(parquet_path)
        records = build_records(parquet_path, args.max_rows)

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for record in records:
        counts[record["source_dataset"]] = counts.get(record["source_dataset"], 0) + 1
    print(json.dumps({"output": str(args.output), "rows_written": len(records), "source_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
