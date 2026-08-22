"""Train a local multilingual email classifier from labelled JSONL data.

This pipeline intentionally uses a pretrained multilingual sentence encoder as a
representation layer and trains task-specific classifiers locally. It does not
claim that the foundation model itself was trained by this project.

Expected JSONL fields:
    id, language, subject, body, intent, urgency, priority

The current 12-case evaluation corpus is deliberately too small for training and
is rejected by the default minimum-size guard. Use a separate, substantially
larger training corpus for model development and keep the evaluation corpus held
out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LABEL_FIELDS = ("intent", "urgency", "priority")
LANGUAGES = ("en", "de", "fr", "es")
DEFAULT_MIN_TRAINING_CASES = 200


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_text = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("id", "language", "subject", "body", *LABEL_FIELDS)
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(f"Line {line_number} missing fields: {', '.join(missing)}")
            if row["language"] not in LANGUAGES:
                raise ValueError(f"Line {line_number}: unsupported language {row['language']!r}")
            text = f"{row['subject']}\n{row['body']}".strip()
            normalized = " ".join(text.lower().split())
            if normalized in seen_text:
                raise ValueError(f"Duplicate email text detected at line {line_number}")
            seen_text.add(normalized)
            rows.append(row)
    if not rows:
        raise ValueError("Training corpus is empty")
    return rows


def text_for(row: Dict[str, Any]) -> str:
    return f"Subject: {row['subject']}\nBody: {row['body']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training JSONL file")
    parser.add_argument("--output", default="artifacts/multilingual_email_model.joblib")
    parser.add_argument("--encoder", default=MODEL_NAME)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-cases", type=int, default=DEFAULT_MIN_TRAINING_CASES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.train))
    if len(rows) < args.min_cases:
        raise ValueError(
            f"Training corpus has {len(rows)} cases; refusing to train with fewer than "
            f"{args.min_cases}. The held-out evaluation corpus must not be used as training data."
        )

    texts = [text_for(row) for row in rows]
    indices = np.arange(len(rows))
    intents = np.array([row["intent"] for row in rows])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=intents,
    )

    encoder = SentenceTransformer(args.encoder)
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    classifiers: Dict[str, LogisticRegression] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for field in LABEL_FIELDS:
        labels = np.array([row[field] for row in rows])
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
        classifier.fit(embeddings[train_idx], labels[train_idx])
        predicted = classifier.predict(embeddings[test_idx])
        classifiers[field] = classifier
        metrics[field] = {
            "accuracy": round(float(accuracy_score(labels[test_idx], predicted)), 4),
            "macro_f1": round(float(f1_score(labels[test_idx], predicted, average="macro", zero_division=0)), 4),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "encoder": args.encoder,
            "classifiers": classifiers,
            "languages": LANGUAGES,
            "label_fields": LABEL_FIELDS,
            "metrics": metrics,
            "training_cases": len(train_idx),
            "validation_cases": len(test_idx),
            "seed": args.seed,
        },
        output,
    )

    print(json.dumps({"model": str(output), "encoder": args.encoder, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
