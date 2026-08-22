"""Prepare the English email-related slice of AmazonScience/MASSIVE.

MASSIVE is a multilingual NLU dataset of voice-assistant utterances, not an
ordinary email corpus. This adapter therefore does NOT pretend that its
examples are ordinary email messages. It extracts only the four email-related
intents from the English (en-US) training split and maps them to an explicit
auxiliary taxonomy:

    email_query        -> INFORMATION
    email_querycontact -> INFORMATION
    email_sendemail    -> REQUEST
    email_addcontact   -> REQUEST

The output is intentionally separate from the Smart Inbox gold evaluation
corpus. No urgency or priority labels are inferred from MASSIVE; doing so
would introduce unverified labels. Source locale, original intent and source
id are retained for provenance.

The adapter downloads the published Parquet artifact directly and then loads
that local file. This avoids both the legacy ``massive.py`` dataset script and
Hugging Face filesystem URL resolution issues in modern ``datasets`` versions.

Example:
    python ml/prepare_massive_email.py --output data/massive_email_auxiliary.jsonl
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict
from urllib.request import Request, urlopen

SOURCE_INTENTS: Dict[str, str] = {
    "email_query": "INFORMATION",
    "email_querycontact": "INFORMATION",
    "email_sendemail": "REQUEST",
    "email_addcontact": "REQUEST",
}

LOCALE = "en-US"
LANGUAGE = "en"
TRAIN_PARQUET_URL = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
    "97cb84b8ee1f104fb75c1f2062e3deb99be62803/en-US/massive-train.parquet"
)


def download_parquet(destination: Path) -> None:
    """Download the published MASSIVE Parquet artifact to a local path."""
    request = Request(TRAIN_PARQUET_URL, headers={"User-Agent": "smart-inbox-ai/1.0"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The datasets package is required. Install the optional ML/data dependencies first."
        ) from exc

    # Download first instead of passing an https:// URL to datasets. Recent
    # datasets releases may resolve remote data through hf:// and fail to find
    # the published Parquet path even though the artifact is publicly available.
    with tempfile.TemporaryDirectory(prefix="smart-inbox-massive-") as temp_dir:
        parquet_path = Path(temp_dir) / "massive-train.parquet"
        print(f"Downloading MASSIVE {LOCALE} training data...")
        try:
            download_parquet(parquet_path)
        except Exception as exc:
            raise SystemExit(f"Failed to download MASSIVE Parquet data: {exc}") from exc

        dataset = load_dataset(
            "parquet",
            data_files={"train": str(parquet_path)},
            split="train",
        )

        rows = []
        for item in dataset:
            source_intent = str(item["intent"])
            if source_intent not in SOURCE_INTENTS:
                continue

            rows.append(
                {
                    "id": f"massive-{LOCALE}-{item['id']}",
                    "language": LANGUAGE,
                    "subject": "",
                    "body": item["utt"],
                    "intent": SOURCE_INTENTS[source_intent],
                    "source_dataset": "AmazonScience/massive",
                    "source_locale": LOCALE,
                    "source_intent": source_intent,
                    "source_split": "train",
                    "source_id": str(item["id"]),
                }
            )

    if not rows:
        raise SystemExit("No matching MASSIVE email-intent examples were found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "language": LANGUAGE,
                "locale": LOCALE,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
