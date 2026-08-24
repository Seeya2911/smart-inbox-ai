from pathlib import Path

from openpyxl import load_workbook

from ml.export_manus_dataset import load_rows, export_xlsx


def test_load_rows_removes_empty_and_exact_duplicates(tmp_path: Path):
    source = tmp_path / "raw.jsonl"
    source.write_text(
        '{"id":"1","subject":"Hello","body":"World","source":"enron"}\n'
        '{"id":"2","subject":" hello ","body":"world","source":"spam_corpus"}\n'
        '{"id":"3","subject":"","body":"","source":"phishing_corpus"}\n'
        '{"id":"4","subject":"Different","body":"Message","source":"phishing_corpus"}\n',
        encoding="utf-8",
    )
    rows, stats = load_rows([source])
    assert len(rows) == 2
    assert stats["input_rows"] == 4
    assert stats["empty_removed"] == 1
    assert stats["exact_duplicates_removed"] == 1


def test_export_contains_only_labeling_columns(tmp_path: Path):
    output = tmp_path / "manus.xlsx"
    export_xlsx(
        [
            {"subject": "A", "body": "B", "source": "enron"},
            {"subject": "C", "body": "D", "source": "spam_corpus"},
        ],
        output,
    )
    workbook = load_workbook(output, read_only=True)
    sheet = workbook["emails"]
    assert list(sheet.iter_rows(values_only=True)) == [
        ("subject", "body", "source"),
        ("A", "B", "enron"),
        ("C", "D", "spam_corpus"),
    ]


def test_export_removes_illegal_control_characters(tmp_path: Path):
    output = tmp_path / "manus.xlsx"
    export_xlsx(
        [
            {
                "subject": "Normal\nsubject",
                "body": "Old Enron text\x01with\x0bcontrol\x1fchars\tkept",
                "source": "enron",
            }
        ],
        output,
    )
    workbook = load_workbook(output, read_only=True)
    values = list(workbook.active.iter_rows(values_only=True))
    assert values[0] == ("subject", "body", "source")
    assert values[1] == (
        "Normal\nsubject",
        "Old Enron textwithcontrolchars\tkept",
        "enron",
    )
