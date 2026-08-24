"""Download real public email corpora into an unlabeled canonical JSONL pool.

This stage intentionally does NOT assign Smart Inbox intent or priority labels.
Source-native labels are preserved as metadata only. The production labels are
created later by the labeling pipeline.

Sources:
- Enron: ``corbt/enron-emails``
- SpamAssassin: upstream public corpus at ``spamassassin.apache.org``
- Additional independent corpora: ``puyang2025/seven-phishing-email-datasets``
  with Enron/Assassin rows excluded because those sources are already ingested.
  The retained sources are TREC-05, TREC-06, TREC-07, CEAS-08, and Ling.

All corpus sources are downloaded through public dataset files rather than the
Hugging Face datasets-server rows API, avoiding rate-limit failures.
"""
from __future__ import annotations

import argparse
import csv
import email
import email.policy
import io
import json
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_DATASET_RESOLVE = "https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets/resolve/main/"
ENRON_DATASET_RESOLVE = "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/"
ENRON_PARQUET_FILES = (
    "train-00000-of-00003.parquet",
    "train-00001-of-00003.parquet",
    "train-00002-of-00003.parquet",
)
PHISHING_DATASET = "puyang2025/seven-phishing-email-datasets"
PHISHING_SOURCE_ALLOWLIST = ("TREC-05", "TREC-06", "TREC-07", "CEAS-08", "Ling")
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_FILES = (
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
    "20050311_spam_2.tar.bz2",
)


@dataclass(frozen=True)
class RawEmailRecord:
    """Unlabeled, provenance-preserving representation of one email."""
    id: str
    subject: str
    body: str
    source: str
    source_example_id: str
    source_split: str
    source_dataset: str = ""
    source_label: str = ""
    is_synthetic: bool = False
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    """Return a bounded delay, honoring Retry-After when supplied."""
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return min(60.0, max(0.5, float(retry_after)))
        except ValueError:
            pass
    return min(60.0, 1.0 * (2**attempt))


def _request_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    *,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    """Fetch one page from Hugging Face's rows API for compatibility/tests."""
    params = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    request = urllib.request.Request(
        f"{HF_ROWS_API}?{params}",
        headers={"User-Agent": "smart-inbox-ai-dataset-ingest/3.0"},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310 - fixed HTTPS API
                payload = json.load(response)
            return [row.get("row", {}) for row in payload.get("rows", [])]
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt >= max_retries:
                raise
            time.sleep(_retry_delay(error, attempt))
    raise RuntimeError("unreachable")


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _subject_from_text(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        if line.lower().startswith("subject:"):
            return line.split(":", 1)[1].strip()
    return ""


def _record_from_common(
    row: dict[str, Any], *, source: str, source_example_id: str, split: str,
    source_dataset: str, source_label: str = "",
) -> RawEmailRecord:
    text = _first_text(row, "text", "body", "email")
    subject = _first_text(row, "subject") or _subject_from_text(text)
    return RawEmailRecord(
        id=f"{source}_{source_example_id}", subject=subject, body=text, source=source,
        source_example_id=source_example_id, source_split=split,
        source_dataset=source_dataset, source_label=source_label,
    )


def normalize_enron(row: dict[str, Any], source_example_id: str, split: str) -> RawEmailRecord:
    return _record_from_common(row, source="enron", source_example_id=source_example_id,
                               split=split, source_dataset="corbt/enron-emails")


def normalize_spamassassin(row: dict[str, Any], source_example_id: str, split: str) -> RawEmailRecord:
    source_label = row.get("label")
    return _record_from_common(row, source="spam_corpus", source_example_id=source_example_id,
                               split=split, source_dataset="talby/spamassassin",
                               source_label="" if source_label is None else str(source_label))


def normalize_phishing_row(row: dict[str, Any], source_example_id: str, split: str) -> RawEmailRecord | None:
    """Normalize one row while excluding duplicate source corpora already ingested."""
    dataset_name = str(row.get("dataset_name", "")).strip()
    if dataset_name not in PHISHING_SOURCE_ALLOWLIST:
        return None
    source_label = row.get("label")
    return _record_from_common(
        row, source="phishing_corpus", source_example_id=f"{dataset_name}:{source_example_id}",
        split=split, source_dataset=dataset_name,
        source_label="" if source_label is None else str(source_label),
    )


def _message_text(raw: bytes) -> tuple[str, str]:
    """Extract a subject and human-readable text from one raw email."""
    message = email.message_from_bytes(raw, policy=email.policy.default)
    subject = str(message.get("Subject", "")).strip()
    plain_parts: list[str] = []
    html_parts: list[str] = []
    parts = message.walk() if message.is_multipart() else (message,)
    for part in parts:
        if part.get_content_disposition() == "attachment":
            continue
        if part.get_content_maintype() != "text":
            continue
        try:
            content = part.get_content()
        except (LookupError, UnicodeError, TypeError):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if part.get_content_subtype() == "plain":
            plain_parts.append(content.strip())
        elif part.get_content_subtype() == "html":
            html_parts.append(content.strip())
    body_parts = plain_parts or html_parts
    return subject, "\n\n".join(body_parts).strip()


def _open_url_with_retries(url: str, *, max_retries: int = 5):
    """Open a public URL with bounded retry handling for transient HTTP failures."""
    request = urllib.request.Request(url, headers={"User-Agent": "smart-inbox-ai-dataset-ingest/3.0"})
    for attempt in range(max_retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=120)  # nosec B310 - fixed HTTPS URLs
        except urllib.error.HTTPError as error:
            if error.code not in {429, 502, 503, 504} or attempt >= max_retries:
                raise
            time.sleep(_retry_delay(error, attempt))
    raise RuntimeError("unreachable")


def _download_to_temp(url: str, suffix: str) -> Path:
    """Download one public dataset file to a temporary local path."""
    response = _open_url_with_retries(url)
    handle = tempfile.NamedTemporaryFile(prefix="smart_inbox_dataset_", suffix=suffix, delete=False)
    path = Path(handle.name)
    try:
        with response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    finally:
        handle.close()
    return path


def iter_enron_direct(*, max_rows: int | None, sleep_seconds: float) -> Iterable[RawEmailRecord]:
    """Yield Enron rows from the public Parquet files without the rows API."""
    if max_rows is None:
        quotas = [None] * len(ENRON_PARQUET_FILES)
    else:
        base, remainder = divmod(max_rows, len(ENRON_PARQUET_FILES))
        quotas = [base + (index < remainder) for index in range(len(ENRON_PARQUET_FILES))]

    emitted = 0
    for filename, quota in zip(ENRON_PARQUET_FILES, quotas):
        if quota == 0:
            continue
        url = urllib.parse.urljoin(ENRON_DATASET_RESOLVE, filename) + "?download=true"
        path = _download_to_temp(url, ".parquet")
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            remaining = quota
            for row_group in range(parquet.num_row_groups):
                if remaining is not None and remaining <= 0:
                    break
                table = parquet.read_row_group(row_group, columns=["message_id", "subject", "body"])
                for row in table.to_pylist():
                    source_id = str(row.get("message_id") or f"{filename}:{emitted}")
                    record = normalize_enron(row, source_id, "train")
                    if not (record.subject.strip() or record.body.strip()):
                        continue
                    yield record
                    emitted += 1
                    if remaining is not None:
                        remaining -= 1
                    if max_rows is not None and emitted >= max_rows:
                        return
        finally:
            path.unlink(missing_ok=True)
        if sleep_seconds:
            time.sleep(sleep_seconds)


def iter_spamassassin_direct(*, max_rows: int | None, sleep_seconds: float) -> Iterable[RawEmailRecord]:
    """Yield SpamAssassin messages directly from the upstream public archives."""
    emitted = 0
    for filename in SPAMASSASSIN_FILES:
        url = urllib.parse.urljoin(SPAMASSASSIN_BASE_URL, filename)
        with _open_url_with_retries(url) as response:
            with tarfile.open(fileobj=response, mode="r|bz2") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    subject, body = _message_text(handle.read())
                    if not (subject or body):
                        continue
                    group = member.name.split("/", 1)[0]
                    label = "ham" if "ham" in group else "spam"
                    yield RawEmailRecord(
                        id=f"spam_corpus_{member.name}", subject=subject, body=body,
                        source="spam_corpus", source_example_id=member.name, source_split="train",
                        source_dataset="talby/spamassassin", source_label=label,
                    )
                    emitted += 1
                    if max_rows is not None and emitted >= max_rows:
                        return
        if sleep_seconds:
            time.sleep(sleep_seconds)


def _phishing_file_url(source_dataset: str) -> str:
    filename = urllib.parse.quote(f"{source_dataset}.csv")
    return urllib.parse.urljoin(HF_DATASET_RESOLVE, f"data_raw/{filename}") + "?download=true"


def _iter_phishing_csv(source_dataset: str, quota: int) -> Iterable[RawEmailRecord]:
    """Stream one source CSV directly from the public dataset file."""
    with _open_url_with_retries(_phishing_file_url(source_dataset)) as response:
        text_stream = io.TextIOWrapper(response, encoding="utf-8-sig", errors="replace", newline="")
        try:
            reader = csv.DictReader(text_stream)
            emitted = 0
            for index, row in enumerate(reader):
                record = normalize_phishing_row(
                    {**row, "dataset_name": source_dataset}, str(index), "train"
                )
                if record is None or not (record.subject.strip() or record.body.strip()):
                    continue
                yield record
                emitted += 1
                if emitted >= quota:
                    break
        finally:
            text_stream.detach()


def iter_phishing_direct(*, max_rows: int | None, sleep_seconds: float) -> Iterable[RawEmailRecord]:
    """Yield a balanced sample from retained phishing source CSVs."""
    if max_rows is None:
        quota = None
    else:
        quota = max(1, (max_rows + len(PHISHING_SOURCE_ALLOWLIST) - 1) // len(PHISHING_SOURCE_ALLOWLIST))
    emitted = 0
    for source_dataset in PHISHING_SOURCE_ALLOWLIST:
        source_quota = quota if quota is not None else 10**9
        for record in _iter_phishing_csv(source_dataset, source_quota):
            yield record
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return
        if sleep_seconds:
            time.sleep(sleep_seconds)


def iter_source(*, dataset: str, config: str, split: str, source: str,
                max_rows: int | None, batch_size: int, sleep_seconds: float) -> Iterable[RawEmailRecord]:
    """Yield records from a configured source, filtering only where required."""
    if source == "enron":
        yield from iter_enron_direct(max_rows=max_rows, sleep_seconds=sleep_seconds)
        return
    if source == "spam_corpus":
        yield from iter_spamassassin_direct(max_rows=max_rows, sleep_seconds=sleep_seconds)
        return
    if source == "phishing_corpus":
        yield from iter_phishing_direct(max_rows=max_rows, sleep_seconds=sleep_seconds)
        return
    raise ValueError(f"Unsupported source: {source}")


def write_jsonl(records: Iterable[RawEmailRecord], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def source_config(source: str) -> tuple[str, str, str]:
    if source == "enron":
        return "corbt/enron-emails", "default", "train"
    if source == "spam_corpus":
        return "talby/spamassassin", "text", "train"
    if source == "phishing_corpus":
        return PHISHING_DATASET, "default", "train"
    raise ValueError(f"Unsupported source: {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download real public email corpora into unlabeled canonical JSONL.")
    parser.add_argument("--source", choices=["enron", "spam_corpus", "phishing_corpus"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be positive when supplied")
    dataset, config, split = source_config(args.source)
    records = iter_source(dataset=dataset, config=config, split=split, source=args.source,
                          max_rows=args.max_rows, batch_size=args.batch_size,
                          sleep_seconds=args.sleep_seconds)
    count = write_jsonl(records, args.output)
    print(json.dumps({"output": str(args.output), "source": args.source, "rows_written": count}, indent=2))


if __name__ == "__main__":
    main()
