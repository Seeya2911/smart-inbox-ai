"""Comprehensive Evaluation CLI for INTENT Models.

Usage:
    python -m ml.evaluate_intent --model artifacts/intent_model.joblib --data artifacts/intent_canonical_dataset.json

Compares:
1. Keyword baseline
2. TF-IDF + Logistic Regression
3. Multilingual transformer embeddings + Logistic Regression

Per project requirements, OpenAI evaluation is NOT run in this change.
Metrics for classes with 0 test examples are excluded from reporting.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ml.intent_classifier import (
    PRETRAINED_MODEL_ID,
    EmbeddingIntentClassifier,
    KeywordIntentClassifier,
    TfidfIntentClassifier,
    load_intent_model,
)
from ml.schema import CanonicalIntentExample


def load_dataset_splits(data_path: Path) -> Tuple[List[CanonicalIntentExample], List[CanonicalIntentExample], List[CanonicalIntentExample], Dict[str, Any]]:
    """Load train, val, and test splits from a JSON dataset artifact or JSONL file."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    text_content = data_path.read_text(encoding="utf-8")
    metadata: Dict[str, Any] = {}

    if data_path.suffix == ".json":
        payload = json.loads(text_content)
        metadata = payload.get("metadata", {})
        splits = payload.get("splits", {})

        train_ex = [CanonicalIntentExample.from_dict(d) for d in splits.get("train", [])]
        val_ex = [CanonicalIntentExample.from_dict(d) for d in splits.get("val", [])]
        test_ex = [CanonicalIntentExample.from_dict(d) for d in splits.get("test", [])]
        return train_ex, val_ex, test_ex, metadata

    # If JSONL format, parse all records as test set
    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    test_ex = [CanonicalIntentExample.from_dict(r) for r in records]
    return [], [], test_ex, metadata


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute accuracy, macro F1, weighted F1, per-class P/R/F1, and confusion matrix.

    Ignores classes that have 0 examples in y_true per requirements.
    """
    if not y_true:
        return {"error": "Empty ground truth labels"}

    # Present target classes in ground truth
    present_classes = sorted(list(set(y_true)))

    if not present_classes:
        return {"error": "No ground truth classes present"}

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=present_classes,
        zero_division=0,
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls_name, p, r, f, sup in zip(present_classes, precision, recall, f1, support):
        per_class[cls_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(sup),
        }

    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": present_classes,
            "matrix": cm.tolist(),
        },
    }


def compute_distribution(examples: List[CanonicalIntentExample], attr: str) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for ex in examples:
        val = str(getattr(ex, attr, "unknown"))
        dist[val] = dist.get(val, 0) + 1
    return dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to canonical split dataset JSON or JSONL file")
    parser.add_argument("--model", help="Optional path to pre-trained intent model joblib artifact")
    parser.add_argument("--output", default="artifacts/intent_eval_results.json", help="Output JSON results path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for baselines")
    args = parser.parse_args()

    data_path = Path(args.data)
    train_ex, val_ex, test_ex, dataset_metadata = load_dataset_splits(data_path)

    if not test_ex:
        raise ValueError("Test set is empty; cannot evaluate models.")

    y_true = [ex.canonical_intent for ex in test_ex]

    eval_results: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_corpus": str(data_path),
        "example_counts": {
            "train": len(train_ex),
            "validation": len(val_ex),
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
        "models": {},
    }

    # 1. Model: Legacy Keyword Baseline
    keyword_clf = KeywordIntentClassifier()
    keyword_preds = keyword_clf.predict(test_ex)
    eval_results["models"]["keyword_baseline"] = {
        "name": "Keyword Baseline (Legacy Rules)",
        "metrics": compute_intent_metrics(y_true, keyword_preds),
    }

    # 2. Model: TF-IDF + Logistic Regression
    tfidf_clf = TfidfIntentClassifier(seed=args.seed)
    if train_ex:
        tfidf_clf.fit(train_ex)
    else:
        tfidf_clf.fit(test_ex)  # fallback if evaluating standalone JSONL
    tfidf_preds = tfidf_clf.predict(test_ex)
    eval_results["models"]["tfidf_logistic_regression"] = {
        "name": "TF-IDF + Logistic Regression",
        "metrics": compute_intent_metrics(y_true, tfidf_preds),
    }

    # 3. Model: Multilingual Transformer Embeddings + Logistic Regression
    if args.model and Path(args.model).is_file():
        loaded_model, model_meta = load_intent_model(args.model)
        transformer_preds = loaded_model.predict(test_ex)
        model_name_str = model_meta.get("model_identifier", PRETRAINED_MODEL_ID)
    else:
        emb_clf = EmbeddingIntentClassifier(model_name=PRETRAINED_MODEL_ID, seed=args.seed)
        if train_ex:
            emb_clf.fit(train_ex)
        else:
            emb_clf.fit(test_ex)
        transformer_preds = emb_clf.predict(test_ex)
        model_name_str = PRETRAINED_MODEL_ID

    eval_results["models"]["transformer_logistic_regression"] = {
        "name": f"Multilingual Transformer ({model_name_str}) + Logistic Regression",
        "metrics": compute_intent_metrics(y_true, transformer_preds),
    }

    # Save results to output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Print comparative report
    print(f"\n=== INTENT EVALUATION REPORT ({data_path.name}) ===")
    print(f"Test Examples: {len(test_ex)} | Classes: {sorted(list(set(y_true)))}\n")
    print(f"{'Model':<60} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 98)

    for m_key, m_info in eval_results["models"].items():
        m_name = m_info["name"]
        m_acc = m_info["metrics"].get("accuracy", 0.0)
        m_f1 = m_info["metrics"].get("macro_f1", 0.0)
        m_wf1 = m_info["metrics"].get("weighted_f1", 0.0)
        print(f"{m_name:<60} | {m_acc:<10.4f} | {m_f1:<10.4f} | {m_wf1:<10.4f}")

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
