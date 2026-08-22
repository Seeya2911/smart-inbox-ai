"""Reproducible Supervised NLP Training Pipeline CLI for INTENT Task.

Usage:
    python -m ml.train_intent --data path/to/dataset.jsonl --output-model artifacts/intent_model.joblib

This CLI ingests raw dataset examples, maps source labels defensibly to the canonical
Smart Inbox intent taxonomy, performs strict quality & duplicate checks, splits data
deterministically, verifies zero split leakage, trains a downstream intent classifier,
evaluates performance, and saves a reproducible model artifact with metadata.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ml.data_quality import check_dataset_integrity
from ml.dataset_splitter import split_intent_dataset
from ml.intent_classifier import (
    PRETRAINED_MODEL_ID,
    EmbeddingIntentClassifier,
    TfidfIntentClassifier,
    save_intent_model,
)
from ml.intent_mapping import map_and_filter_dataset
from ml.schema import CanonicalIntentExample


def load_raw_jsonl(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load raw JSONL rows from one or multiple file paths."""
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset input file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no} in {path}") from exc
    if not records:
        raise ValueError("No records found in provided input data path(s)")
    return records


def get_environment_metadata() -> Dict[str, str]:
    """Capture Python and package versions for reproducibility."""
    import joblib
    import sklearn

    pkg_versions = {
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
    }

    try:
        import sentence_transformers

        pkg_versions["sentence-transformers"] = sentence_transformers.__version__
    except ImportError:
        pkg_versions["sentence-transformers"] = "not-installed"

    return pkg_versions


def compute_distribution(examples: List[CanonicalIntentExample], attr: str) -> Dict[str, int]:
    """Compute distribution count for a given CanonicalIntentExample attribute."""
    dist: Dict[str, int] = {}
    for ex in examples:
        val = str(getattr(ex, attr, "unknown"))
        dist[val] = dist.get(val, 0) + 1
    return dist


def compute_simple_metrics(classifier: Any, examples: List[CanonicalIntentExample]) -> Dict[str, float]:
    """Compute accuracy and macro F1 on a set of examples."""
    from sklearn.metrics import accuracy_score, f1_score

    if not examples:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    y_true = [ex.canonical_intent for ex in examples]
    y_pred = classifier.predict(examples)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, nargs="+", help="Input dataset JSONL file(s)")
    parser.add_argument("--classifier", choices=("embedding", "tfidf"), default="embedding", help="Classifier type")
    parser.add_argument("--output-model", default="artifacts/intent_model.joblib", help="Output model joblib path")
    parser.add_argument("--output-dataset", default="artifacts/intent_canonical_dataset.json", help="Output split dataset path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--min-cases", type=int, default=10, help="Minimum mapped cases required")
    parser.add_argument("--allow-internal-duplicates", action="store_true", help="Allow internal duplicate text within raw dataset")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.data]
    raw_records = load_raw_jsonl(input_paths)

    # 1. Label Mapping & Exclusion Recording
    valid_examples, exclusions, mapping_summary = map_and_filter_dataset(raw_records)

    if len(valid_examples) < args.min_cases:
        raise ValueError(
            f"Dataset contains only {len(valid_examples)} valid mapped cases; "
            f"refusing to train with fewer than {args.min_cases} cases."
        )

    # 2. Data Quality & Cleanliness Check
    quality_summary = check_dataset_integrity(valid_examples, allow_internal_duplicates=args.allow_internal_duplicates)

    # 3. Deterministic Splitting & Leakage Protection
    train_ex, val_ex, test_ex = split_intent_dataset(
        valid_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # 4. Model Training
    if args.classifier == "embedding":
        clf = EmbeddingIntentClassifier(model_name=PRETRAINED_MODEL_ID, seed=args.seed)
        model_identifier = PRETRAINED_MODEL_ID
    else:
        clf = TfidfIntentClassifier(seed=args.seed)
        model_identifier = "tfidf-logistic-regression"

    clf.fit(train_ex)

    # 5. Evaluate Validation & Test Splits
    val_metrics = compute_simple_metrics(clf, val_ex)
    test_metrics = compute_simple_metrics(clf, test_ex)

    # 6. Build Reproducibility Metadata
    metadata: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": "intent",
        "classifier_type": args.classifier,
        "model_identifier": model_identifier,
        "seed": args.seed,
        "environment": get_environment_metadata(),
        "training_config": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "min_cases": args.min_cases,
            "C": 1.0,
            "class_weight": "balanced",
        },
        "dataset_summary": {
            "input_files": [str(p) for p in input_paths],
            "total_raw_records": len(raw_records),
            "mapped_valid_examples": len(valid_examples),
            "excluded_examples": len(exclusions),
            "mapping_stats": mapping_summary,
            "quality_stats": quality_summary,
        },
        "split_counts": {
            "train": len(train_ex),
            "val": len(val_ex),
            "test": len(test_ex),
        },
        "class_distributions": {
            "train": compute_distribution(train_ex, "canonical_intent"),
            "val": compute_distribution(val_ex, "canonical_intent"),
            "test": compute_distribution(test_ex, "canonical_intent"),
        },
        "language_distributions": {
            "train": compute_distribution(train_ex, "language"),
            "val": compute_distribution(val_ex, "language"),
            "test": compute_distribution(test_ex, "language"),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # 7. Save Model Artifact & Dataset Output
    output_model_path = Path(args.output_model)
    save_intent_model(clf, str(output_model_path), metadata)

    output_ds_path = Path(args.output_dataset)
    output_ds_path.parent.mkdir(parents=True, exist_ok=True)
    with output_ds_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": metadata,
                "splits": {
                    "train": [ex.to_dict() for ex in train_ex],
                    "val": [ex.to_dict() for ex in val_ex],
                    "test": [ex.to_dict() for ex in test_ex],
                },
                "exclusions": [ex.to_dict() for ex in exclusions],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    summary_out = {
        "status": "success",
        "model_artifact": str(output_model_path),
        "dataset_artifact": str(output_ds_path),
        "model_identifier": model_identifier,
        "splits": metadata["split_counts"],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if len(valid_examples) < 50:
        summary_out["pipeline_note"] = (
            f"The training dataset contains only {len(valid_examples)} cases. "
            "Small development datasets/fixtures are used exclusively for pipeline mechanics validation and MUST NOT be presented as real-world model benchmarks."
        )
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
