from __future__ import annotations

import email
import io
import json
import urllib.error

from ml import email_dataset_ingest as ingest
from ml.schema import format_namespaced_id


def test_normalize_enron_preserves_provenance_and_subject() -> None:
    record = ingest.normalize_enron(
        {"text": "Subject: Project update\n\nThe work is progressing."}, "42", "train"
    )
    assert record.id == "enron_42"
    assert record.source == "enron"
    assert record.source_example_id == "42"
    assert record.source_dataset == "corbt/enron-emails"
    assert record.subject == "Project update"
    assert record.is_synthetic is False


def test_normalize_spamassassin_preserves_source_label_without_mapping_it() -> None:
    record = ingest.normalize_spamassassin(
        {"text": "Subject: Special offer\n\nBuy now.", "label": "spam"}, "7", "train"
    )
    assert record.source == "spam_corpus"
    assert record.source_dataset == "talby/spamassassin"
    assert record.source_label == "spam"
    assert record.subject == "Special offer"


def test_message_text_extracts_subject_and_plain_body() -> None:
    message = email.message.EmailMessage()
    message["Subject"] = "Account notice"
    message.set_content("Please review this message.")
    subject, body = ingest._message_text(message.as_bytes())
    assert subject == "Account notice"
    assert body == "Please review this message."


def test_iter_source_uses_direct_enron_path(monkeypatch) -> None:
    expected = ingest.RawEmailRecord(
        "enron_42", "Project update", "The work is progressing.", "enron", "42", "train", "corbt/enron-emails"
    )

    def fake_direct(*, max_rows: int | None, sleep_seconds: float):
        assert max_rows == 4000
        assert sleep_seconds == 0
        yield expected

    def fail_rows(*args, **kwargs):
        raise AssertionError("Enron must not use the Hugging Face rows API")

    monkeypatch.setattr(ingest, "iter_enron_direct", fake_direct)
    monkeypatch.setattr(ingest, "_request_rows", fail_rows)
    rows = list(
        ingest.iter_source(
            dataset="corbt/enron-emails", config="default", split="train", source="enron",
            max_rows=4000, batch_size=100, sleep_seconds=0,
        )
    )
    assert rows == [expected]


def test_iter_source_uses_direct_spamassassin_path(monkeypatch) -> None:
    expected = ingest.RawEmailRecord(
        "spam_corpus_mail-1", "Offer", "Body", "spam_corpus", "mail-1", "train", "talby/spamassassin", "spam"
    )
    calls: list[tuple[int | None, float]] = []

    def fake_direct(*, max_rows: int | None, sleep_seconds: float):
        calls.append((max_rows, sleep_seconds))
        yield expected

    def fail_rows(*args, **kwargs):
        raise AssertionError("SpamAssassin must not use the Hugging Face rows API")

    monkeypatch.setattr(ingest, "iter_spamassassin_direct", fake_direct)
    monkeypatch.setattr(ingest, "_request_rows", fail_rows)
    rows = list(
        ingest.iter_source(
            dataset="talby/spamassassin", config="text", split="train", source="spam_corpus",
            max_rows=3000, batch_size=100, sleep_seconds=0,
        )
    )
    assert rows == [expected]
    assert calls == [(3000, 0)]


def test_iter_source_uses_direct_phishing_path(monkeypatch) -> None:
    expected = ingest.RawEmailRecord(
        "phishing_corpus_TREC-05:1", "Notice", "Body", "phishing_corpus", "TREC-05:1", "train", "TREC-05", "1"
    )

    def fake_direct(*, max_rows: int | None, sleep_seconds: float):
        assert max_rows == 3000
        assert sleep_seconds == 0
        yield expected

    def fail_rows(*args, **kwargs):
        raise AssertionError("Phishing corpus must not use the Hugging Face rows API")

    monkeypatch.setattr(ingest, "iter_phishing_direct", fake_direct)
    monkeypatch.setattr(ingest, "_request_rows", fail_rows)
    rows = list(
        ingest.iter_source(
            dataset=ingest.PHISHING_DATASET, config="default", split="train", source="phishing_corpus",
            max_rows=3000, batch_size=100, sleep_seconds=0,
        )
    )
    assert rows == [expected]


def test_iter_phishing_csv_normalizes_source_and_native_label(monkeypatch) -> None:
    csv_bytes = (
        "subject,text,label,dataset_name\n"
        'Security notice,"Please review your account",1,TREC-05\n'
        'Routine note,"See you tomorrow",0,TREC-05\n'
    ).encode()

    class Response(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    monkeypatch.setattr(ingest, "_open_url_with_retries", lambda url: Response(csv_bytes))
    rows = list(ingest._iter_phishing_csv("TREC-05", 2))
    assert [row.subject for row in rows] == ["Security notice", "Routine note"]
    assert [row.source_dataset for row in rows] == ["TREC-05", "TREC-05"]
    assert [row.source_label for row in rows] == ["1", "0"]


def test_normalize_phishing_filters_already_ingested_sources() -> None:
    duplicate = ingest.normalize_phishing_row(
        {"dataset_name": "Enron", "text": "Subject: Existing"}, "1", "train"
    )
    assert duplicate is None
    record = ingest.normalize_phishing_row(
        {"dataset_name": "CEAS-08", "subject": "Security notice", "text": "Please review this notice", "label": 1},
        "9", "train",
    )
    assert record is not None
    assert record.id == "phishing_corpus_CEAS-08:9"
    assert record.source == "phishing_corpus"
    assert record.source_dataset == "CEAS-08"
    assert record.source_label == "1"


def test_write_jsonl_keeps_raw_records_unlabeled(tmp_path) -> None:
    output = tmp_path / "emails.jsonl"
    records = [ingest.RawEmailRecord("enron_1", "Hello", "Body", "enron", "1", "train", "corbt/enron-emails")]
    assert ingest.write_jsonl(records, output) == 1
    payload = json.loads(output.read_text(encoding="utf-8").strip())
    assert payload["id"] == "enron_1"
    assert "intent" not in payload
    assert "priority" not in payload


def test_request_rows_retries_http_429(monkeypatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return b'{"rows":[{"row":{"id":"1"}}]}'

    def fake_urlopen(request, timeout=60):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "2"}, None)
        return Response()

    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)
    rows = ingest._request_rows("example", "default", "train", 0, 1, max_retries=2)
    assert rows == [{"id": "1"}]
    assert attempts == 2
    assert sleeps == [2.0]


def test_phishing_namespace_is_supported() -> None:
    assert format_namespaced_id("phishing_corpus", "CEAS-08:9") == "phishing_CEAS-08:9"
