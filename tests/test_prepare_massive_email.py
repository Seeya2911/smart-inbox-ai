import pytest

from ml.prepare_massive_email import deduplicate_rows


def make_row(source_id: str, body: str, intent: str = "INFORMATION") -> dict[str, str]:
    return {
        "id": f"massive-en-US-{source_id}",
        "body": body,
        "intent": intent,
        "source_id": source_id,
        "source_dataset": "AmazonScience/massive",
    }


def test_deduplicate_rows_removes_known_massive_normalized_duplicates():
    duplicate_pairs = [
        ("do i have any new email", "15782", "16757"),
        ("any new emails", "15783", "16488"),
        ("do i have any new emails", "15788", "16915"),
        ("any new email", "16260", "17162"),
    ]
    rows = [
        row
        for text, first_id, second_id in duplicate_pairs
        for row in (make_row(first_id, text), make_row(second_id, f"  {text.upper()}  "))
    ]
    rows.append(make_row("20000", "send the report to alex", "REQUEST"))

    deduplicated, duplicates_removed = deduplicate_rows(rows)

    assert duplicates_removed == 4
    assert [row["source_id"] for row in deduplicated] == ["15782", "15783", "15788", "16260", "20000"]
    assert [row["body"] for row in deduplicated[:4]] == [pair[0] for pair in duplicate_pairs]


def test_deduplicate_rows_rejects_conflicting_labels():
    rows = [
        make_row("1", "do i have any new email", "INFORMATION"),
        make_row("2", "DO I HAVE ANY NEW EMAIL", "REQUEST"),
    ]

    with pytest.raises(ValueError, match="Conflicting mapped labels"):
        deduplicate_rows(rows)
