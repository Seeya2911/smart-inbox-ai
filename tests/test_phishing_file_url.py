from ml import email_dataset_ingest as ingest


def test_phishing_file_url_uses_data_raw_directory() -> None:
    url = ingest._phishing_file_url("TREC-05")
    assert url.startswith(
        "https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets/resolve/main/data_raw/"
    )
    assert url.endswith("TREC-05.csv?download=true")
