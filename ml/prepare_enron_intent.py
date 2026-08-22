"""Prepare the manually verified Enron intent dataset as auxiliary training data.

The source provides a binary action/response intent label. This script preserves
that label as ``action_intent`` instead of inventing urgency or priority labels.
The generated file is an auxiliary artifact and should not be treated as the
final Smart Inbox three-task training corpus.

The raw source data is downloaded at runtime and is deliberately not committed
because it originates from the public Enron email corpus. See
``docs/dataset_protocol.md`` for provenance and privacy guidance.

Example:
    python ml/prepare_enron_intent.py --output data/enron_intent_auxiliary.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

SOURCE_REPO = "https://raw.githubusercontent.com/Charlie9/enron_intent_dataset_verified/master"
FILES = {"negative": "intent_neg", "positive": "intent_pos"}


def download_lines(url: str) -> list[str]:
    with urlopen(url, timeout=30) as response:  # nosec B310 - fixed HTTPS source above
        text = response.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    rows = []
    seen = set()
    for label, filename in FILES.items():
        url = f"{SOURCE_REPO}/{filename}"
        for index, text in enumerate(download_lines(url)):
            normalized = " ".join(text.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            rows.append(
                {
                    "id": f"enron-intent-{label}-{index}",
                    "language": "en",
                    "subject": "",
                    "body": text,
                    "action_intent": "ACTION_REQUIRED" if label == "positive" else "NO_ACTION_REQUIRED",
                    "source_dataset": "Charlie9/enron_intent_dataset_verified",
                    "source_file": filename,
                    "source_split": "unspecified",
                    "source_id": str(index),
                    "data_tier": "public_email_auxiliary",
                }
            )

    if not rows:
        raise SystemExit("No Enron intent examples were downloaded.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in rows:
        counts[row["action_intent"]] = counts.get(row["action_intent"], 0) + 1
    print(json.dumps({"output": str(output), "rows": len(rows), "labels": counts}, indent=2))


if __name__ == "__main__":
    main()
