import json
from pathlib import Path

from ml.corpus_coverage import analyze_corpora


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_coverage_reports_sources_labels_and_cross_source_duplicates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"source": "enron", "subject": "Meeting", "body": "Discuss budget", "language": "en"},
            {"source": "spam_corpus", "subject": "Meeting", "body": "Discuss budget", "language": "en", "source_label": "spam"},
            {"source": "phishing_corpus", "subject": "Password", "body": "Reset now", "language": "en", "source_label": "phishing"},
            {"source": "phishing_corpus", "subject": "", "body": "", "language": "en", "source_label": "benign"},
        ],
    )

    report = analyze_corpora([corpus])

    assert report["total_rows"] == 4
    assert report["source_counts"] == {"enron": 1, "phishing_corpus": 2, "spam_corpus": 1}
    assert report["source_native_labels"] == {
        "phishing_corpus": {"benign": 1, "phishing": 1},
        "spam_corpus": {"spam": 1},
    }
    assert report["language_counts"] == {"en": 4}
    assert report["empty_content_rows"] == 1
    assert report["cross_source_exact_duplicate_groups"] == 1
    assert report["exact_duplicate_groups"] == 1


def test_coverage_does_not_treat_different_whitespace_as_distinct(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"source": "enron", "subject": "Hello", "body": "A   B"},
            {"source": "enron", "subject": "hello", "body": "A B"},
        ],
    )

    report = analyze_corpora([corpus])

    assert report["unique_normalized_content"] == 1
    assert report["exact_duplicate_groups"] == 1
    assert report["cross_source_exact_duplicate_groups"] == 0
