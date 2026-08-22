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

The adapter uses the published Parquet file directly instead of the legacy
``massive.py`` dataset-loading script. This is required by modern versions of
the Hugging Face ``datasets`` package, which no longer execute dataset scripts.

The published Parquet files store the ``intent`` ClassLabel as its numeric
class id. The ids below follow the official MASSIVE intent ordering, so the
adapter explicitly converts those ids back to their intent names before
filtering. String intent names are also accepted for compatibility.

Example:
    python ml/prepare_massive_email.py --output data/massive_email_auxiliary.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

SOURCE_INTENTS: Dict[str, str] = {
    "email_query": "INFORMATION",
    "email_querycontact": "INFORMATION",
    "email_sendemail": "REQUEST",
    "email_addcontact": "REQUEST",
}

# MASSIVE's official _INTENTS ordering. The Parquet ClassLabel values for the
# four email intents are therefore 15, 17, 33 and 44 respectively.
INTENT_ID_TO_NAME: Dict[int, str] = {
    15: "email_addcontact",
    17: "email_querycontact",
    33: "email_sendemail",
    44: "email_query",
}

LOCALE = "en-US"
LANGUAGE = "en"
TRAIN_PARQUET_URL = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/main/"
    "en-US/massive-train.parquet"
)


def normalize_source_intent(value: object) -> str:
    """Return the MASSIVE intent name for either a ClassLabel id or name."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return INTENT_ID_TO_NAME.get(value, "")

    text = str(value).strip()
    if text in SOURCE_INTENTS:
        return text

    try:
        return INTENT_ID_TO_NAME.get(int(text), "")
    except ValueError:
        return ""


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

    # Load the published Parquet artifact directly. This avoids the legacy
    # AmazonScience/massive dataset script, which datasets >= 4 no longer supports.
    dataset = load_dataset(
        "parquet",
        data_files={"train": TRAIN_PARQUET_URL},
        split="train",
    )

    rows = []
    for item in dataset:
        source_intent = normalize_source_intent(item["intent"])
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
