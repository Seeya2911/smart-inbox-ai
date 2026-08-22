"""Run inference with a trained multilingual Smart Inbox classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", required=True)
    args = parser.parse_args()

    artifact = joblib.load(Path(args.model))
    encoder = SentenceTransformer(artifact["encoder"])
    text = f"Subject: {args.subject}\nBody: {args.body}"
    embedding = encoder.encode([text], normalize_embeddings=True)

    result: Dict[str, str] = {}
    for field, classifier in artifact["classifiers"].items():
        result[field] = str(classifier.predict(embedding)[0])

    print(result)


if __name__ == "__main__":
    main()
