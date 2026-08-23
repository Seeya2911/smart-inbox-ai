from __future__ import annotations

import json

from ml import email_dataset_ingest as ingest


def test_normalize_enron_preserves_provenance_and_subject() -> None:
    record = ingest.normalize_enron(
        {"text": "Subject: Project update\n\nThe work is progressing."},
        "42",
        "train",
    )
    assert record.id == "enron-42"
    assert record.source == "enron"
    assert record.source_example_id == "42"
    assert record.subject == "Project update"
    assert record.is_synthetic is False


def test_normalize_spamassassin_uses_text_when_subject_field_missing() -> None:
    record = ingest.normalize_spamassassin(
        {"text": "Subject: Special offer\n\nBuy now."},
        "7",
        "train",
    )
    assert record.source == "spam_corpus"
    assert record.subject == "Special offer"
    assert record.body.startswith("Subject: Special offer")


def test_write_jsonl_skips_empty_records(tmp_path) -> None:
    output = tmp_path / "emails.jsonl"
    records = [
        ingest.RawEmailRecord("1", "Hello", "Body", "enron", "1", "train"),
        ingest.RawEmailRecord("2", "", "", "enron", "2", "train"),
    ]
    assert ingest.write_jsonl(records, output) == 1
    payload = json.loads(output.read_text(encoding="utf-8").strip())
    assert payload["id"] == "1"
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
