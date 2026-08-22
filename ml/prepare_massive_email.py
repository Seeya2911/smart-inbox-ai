"""Prepare the email-related slice of AmazonScience/MASSIVE for auxiliary intent training.

MASSIVE is a multilingual NLU dataset of voice-assistant utterances, not an email
corpus. This adapter therefore does NOT pretend that its examples are ordinary
email messages. It extracts only the four email-related intents and maps them to
an explicit auxiliary taxonomy:

    email_query        -> INFORMATION
    email_querycontact -> INFORMATION
    email_sendemail    -> REQUEST
    email_addcontact   -> REQUEST

The output is intentionally separate from the Smart Inbox gold evaluation corpus.
No urgency or priority labels are inferred from MASSIVE; doing so would introduce
unverified labels. The source locale, original intent and source id are retained
for provenance.

Example:
    python ml/prepare_massive_email.py --output data/massive_email_auxiliary.jsonl

The script downloads MASSIVE through the Hugging Face datasets library at runtime.
The dataset is CC BY 4.0; see the dataset card and project data-provenance docs.
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

LOCALES = {"en-US": "en", "de-DE": "de", "fr-FR": "fr", "es-ES": "es"}


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

    rows = []
    for locale, language in LOCALES.items():
        dataset = load_dataset("AmazonScience/massive", locale, split="train")
        for item in dataset:
            source_intent = item["intent"]
            if source_intent not in SOURCE_INTENTS:
                continue
            rows.append(
                {
                    "id": f"massive-{locale}-{item['id']}",
                    "language": language,
                    "subject": "",
                    "body": item["utt"],
                    "intent": SOURCE_INTENTS[source_intent],
                    "source_dataset": "AmazonScience/massive",
                    "source_locale": locale,
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

    print(json.dumps({"output": str(output), "rows": len(rows), "languages": sorted(LOCALES.values())}, indent=2))


if __name__ == "__main__":
    main()
