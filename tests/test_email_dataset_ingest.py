from __future__ import annotations

import json

from ml import email_dataset_ingest as ingest
from ml.schema import format_namespaced_id


def test_normalize_enron_preserves_provenance_and_subject() -> None:
    record = ingest.normalize_enron(
        {"text": "Subject: Project update\n\nThe work is progressing."},
        "42",
        "train",
    )
    assert record.id == "enron_42"
    assert record.source == "enron"
    assert record.source_example_id == "42"
    assert record.source_dataset == "corbt/enron-emails"
    assert record.subject == "Project update"
    assert record.is_synthetic is False


def test_normalize_spamassassin_preserves_source_label_without_mapping_it() -> None:
    record = ingest.normalize_spamassassin(
        {"text": "Subject: Special offer\n\nBuy now.", "label": "spam"},
        "7",
        "train",
    )
    assert record.source == "spam_corpus"
    assert record.source_dataset == "talby/spamassassin"
    assert record.source_label == "spam"
    assert record.subject == "Special offer"


def test_normalize_phishing_filters_already_ingested_sources() -> None:
    duplicate = ingest.normalize_phishing_row(
        {"dataset_name": "Enron", "text": "Subject: Existing"}, "1", "train"
    )
    assert duplicate is None

    record = ingest.normalize_phishing_row(
        {"dataset_name": "CEAS-08", "subject": "Security notice", "text": "Please review this notice", "label": 1},
        "9",
        "train",
    )
    assert record is not None
    assert record.id == "phishing_corpus_CEAS-08:9"
    assert record.source == "phishing_corpus"
    assert record.source_dataset == "CEAS-08"
    assert record.source_label == "1"


def test_write_jsonl_keeps_raw_records_unlabeled(tmp_path) -> None:
    output = tmp_path / "emails.jsonl"
    records = [
        ingest.RawEmailRecord("enron_1", "Hello", "Body", "enron", "1", "train", "corbt/enron-emails"),
    ]
    assert ingest.write_jsonl(records, output) == 1
    payload = json.loads(output.read_text(encoding="utf-8").strip())
    assert payload["id"] == "enron_1"
    assert "intent" not in payload
    assert "priority" not in payload


def test_iter_source_honors_max_rows(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_request(dataset: str, config: str, split: str, offset: int, length: int):
        calls.append((offset, length))
        return [{"id": str(offset), "text": f"Subject: S{offset}\nbody"}]

    monkeypatch.setattr(ingest, "_request_rows", fake_request)
    rows = list(
        ingest.iter_source(
            dataset="example",
            config="default",
            split="train",
            source="enron",
            max_rows=2,
            batch_size=1,
            sleep_seconds=0,
        )
    )
    assert len(rows) == 2
    assert calls == [(0, 1), (1, 1)]


def test_phishing_namespace_is_supported() -> None:
    assert format_namespaced_id("phishing_corpus", "CEAS-08:9") == "phishing_CEAS-08:9"
