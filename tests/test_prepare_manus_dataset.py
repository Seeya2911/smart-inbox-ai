import json
from pathlib import Path

from openpyxl import load_workbook

from ml.prepare_manus_dataset import prepare_records, write_batches


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_prepare_records_removes_empty_and_exact_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    _write_jsonl(
        source,
        [
            {"id": "enron_1", "subject": "Hello", "body": "Discuss budget", "source": "enron"},
            {"id": "spam_1", "subject": "Hello", "body": "Discuss budget", "source": "spam_corpus"},
            {"id": "enron_2", "subject": "", "body": "", "source": "enron"},
        ],
    )

    records, stats = prepare_records([source])

    assert len(records) == 1
    assert records[0]["id"] == "enron_1"
    assert stats == {"rows_read": 3, "empty_removed": 1, "exact_duplicates_removed": 1}


def test_write_batches_contains_only_manus_input_columns(tmp_path: Path) -> None:
    records = [
        {"id": "enron_1", "subject": "A", "body": "Body A", "source": "enron"},
        {"id": "spam_1", "subject": "B", "body": "Body B", "source": "spam_corpus"},
        {"id": "phishing_1", "subject": "C", "body": "Body C", "source": "phishing_corpus"},
    ]

    manifest = write_batches(records, tmp_path, batch_size=2)

    first = load_workbook(tmp_path / "manus_batch_0001.xlsx", read_only=True).active
    assert list(first.values) == [
        ("subject", "body", "source"),
        ("A", "Body A", "enron"),
        ("B", "Body B", "spam_corpus"),
    ]
    assert len(manifest) == 3
    assert manifest[0] == {"batch": "manus_batch_0001.xlsx", "excel_row": 2, "id": "enron_1", "source": "enron"}
    assert manifest[-1]["batch"] == "manus_batch_0002.xlsx"
