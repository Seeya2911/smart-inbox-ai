from __future__

import pyarrow as pa
import pyarrow.parquet as pq

from ml import phishing_parquet_ingest as ingest


def test_build_records_filters_sources_and_balances_sample(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "text": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "subject": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "label": [0, 1, 0, 1, 0, 1, 1, 0, 1],
            "dataset_name": ["TREC-05", "TREC-05", "TREC-06", "CEAS-08", "Ling", "Enron", "Assassin", "TREC-06", "TREC-07"],
        }
    )
    pq.write_table(table, path)

    rows = ingest.build_records(path, 5)
    assert len(rows) == 5
    assert {row["source"] for row in rows} == {"phishing_corpus"}
    assert {row["source_dataset"] for row in rows} == {"TREC-05", "TREC-06", "TREC-07", "CEAS-08", "Ling"}
    assert all("intent" not in row and "priority" not in row for row in rows)
    assert rows[0]["source_label"] in {"0", "1"}


def test_download_url_is_verified_parquet_artifact() -> None:
    assert ingest.DATASET_URL.endswith("/train.parquet?download=true")
    assert "datasets-server.huggingface.co/rows" not in ingest.DATASET_URL
