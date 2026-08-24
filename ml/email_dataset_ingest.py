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

The Hugging Face datasets-server rows API is used for Enron and the phishing
corpora. SpamAssassin is downloaded directly from its upstream public corpus so
this workflow does not depend on Hugging Face's on-demand rows API for that
source.
"""
from __future__ import annotations

import argparse
import email
import email.policy
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
PHISHING_DATASET = "puyang2025/seven-phishing-email-datasets"
PHISHING_SOURCE_ALLOWLIST = {"TREC-05", "TREC-06", "TREC-07", "CEAS-08", "Ling"}
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
    """Fetch one page from Hugging Face, retrying transient 429 responses."""
    params = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    request = urllib.request.Request(
        f"{HF_ROWS_API}?{params}",
        headers={"User-Agent": "smart-inbox-ai-dataset-ingest/2.0"},
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
    """Open a public URL with bounded retry handling for transient 429s."""
    request = urllib.request.Request(url, headers={"User-Agent": "smart-inbox-ai-dataset-ingest/2.0"})
    for attempt in range(max_retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=120)  # nosec B310 - fixed HTTPS URLs
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt >= max_retries:
                raise
            time.sleep(_retry_delay(error, attempt))
    raise RuntimeError("unreachable")


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
                        id=f"spam_corpus_{member.name}",
                        subject=subject,
                        body=body,
                        source="spam_corpus",
                        source_example_id=member.name,
                        source_split="train",
                        source_dataset="talby/spamassassin",
                        source_label=label,
                    )
                    emitted += 1
                    if max_rows is not None and emitted >= max_rows:
                        return
        if sleep_seconds:
            time.sleep(sleep_seconds)


def iter_source(*, dataset: str, config: str, split: str, source: str,
                max_rows: int | None, batch_size: int, sleep_seconds: float) -> Iterable[RawEmailRecord]:
    """Yield records from a configured source, filtering only where required."""
    if source == "spam_corpus":
        yield from iter_spamassassin_direct(max_rows=max_rows, sleep_seconds=sleep_seconds)
        return

    offset = 0
    emitted = 0
    while max_rows is None or emitted < max_rows:
        length = batch_size if max_rows is None else min(batch_size, max_rows - emitted)
        rows = _request_rows(dataset, config, split, offset, length)
        if not rows:
            break
        for index, row in enumerate(rows):
            source_id = str(row.get("id", offset + index))
            if source == "enron":
                record = normalize_enron(row, source_id, split)
            elif source == "phishing_corpus":
                record = normalize_phishing_row(row, source_id, split)
                if record is None:
                    continue
            else:
                raise ValueError(f"Unsupported source: {source}")
            if not (record.subject.strip() or record.body.strip()):
                continue
            yield record
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
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
