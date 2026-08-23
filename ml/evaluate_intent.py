"""Comprehensive Evaluation CLI for INTENT Models.

Usage:
    python -m ml.evaluate_intent --data artifacts/intent_canonical_dataset.json --model artifacts/intent_model.joblib

Compares:
1. Keyword baseline (Rule-based, no fitting required)
2. TF-IDF + Logistic Regression (Requires genuine training split or loaded TF-IDF model)
3. Multilingual transformer embeddings + Logistic Regression (Requires genuine training split or loaded embedding model)

METHODOLOGICAL REQUIREMENT:
Trainable models MUST receive a genuine training split (or pre-trained model checkpoint).
Evaluating on test-only data WITHOUT a training split or pre-trained model FAILS LOUDLY with a ValueError.
Fitting trainable models on validation or test data is STRICTLY PROHIBITED.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ml.intent_classifier import (
    PRETRAINED_MODEL_ID,
    BaseIntentClassifier,
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

    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    from ml.intent_mapping import map_and_filter_dataset

    valid_ex, _, mapping_stats = map_and_filter_dataset(records)
    metadata["mapping_stats"] = mapping_stats
    return [], [], valid_ex, metadata


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute accuracy, macro F1, weighted F1, per-class P/R/F1, and confusion matrix."""
    if not y_true:
        return {"error": "Empty ground truth labels"}

    present_classes = sorted(list(set(y_true)))
    if not present_classes:
        return {"error": "No ground truth classes present"}

    acc = float(accuracy_score(y_true, y_pred))

    if len(present_classes) == 1:
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(acc, 4),
            "weighted_f1": round(acc, 4),
            "per_class": {present_classes[0]: {"precision": round(acc, 4), "recall": round(acc, 4), "f1": round(acc, 4), "support": len(y_true)}},
            "confusion_matrix": {"labels": present_classes, "matrix": [[len(y_true)]]},
            "note": "Evaluation test set contains only 1 intent class; macro/weighted F1 reflect single-class accuracy.",
        }

    macro_f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=present_classes, average="weighted", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=present_classes, zero_division=0)

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls_name, p, r, f, sup in zip(present_classes, precision, recall, f1, support):
        per_class[cls_name] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "support": int(sup)}

    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": present_classes, "matrix": cm.tolist()},
    }


def compute_distribution(examples: List[CanonicalIntentExample], attr: str) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for ex in examples:
        val = str(getattr(ex, attr, "unknown"))
        dist[val] = dist.get(val, 0) + 1
    return dist


def _evaluate_classifier(clf: BaseIntentClassifier, test_ex: List[CanonicalIntentExample], y_true: List[str]) -> Dict[str, Any]:
    """Predict on the untouched test split and return standardized metrics."""
    return compute_intent_metrics(y_true, clf.predict(test_ex))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to canonical split dataset JSON or JSONL file")
    parser.add_argument("--model", help="Optional path to a pre-trained intent model joblib artifact")
    parser.add_argument("--output", default="artifacts/intent_eval_results.json", help="Output JSON results path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for baselines")
    args = parser.parse_args()

    data_path = Path(args.data)
    train_ex, val_ex, test_ex, dataset_metadata = load_dataset_splits(data_path)
    if not test_ex:
        raise ValueError("Test split is empty; cannot run evaluation.")

    y_true = [ex.canonical_intent for ex in test_ex]
    eval_results: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_corpus": str(data_path),
        "example_counts": {"train": len(train_ex), "validation": len(val_ex), "test": len(test_ex)},
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

    keyword_clf = KeywordIntentClassifier()
    eval_results["models"]["keyword_baseline"] = {
        "name": "Keyword Baseline (Legacy Rules)",
        "metrics": _evaluate_classifier(keyword_clf, test_ex, y_true),
    }

    loaded_model: BaseIntentClassifier | None = None
    model_meta: Dict[str, Any] = {}
    if args.model and Path(args.model).is_file():
        loaded_model, model_meta = load_intent_model(args.model)

    # A loaded artifact is used only for the classifier it actually contains.
    # Never label a TF-IDF model as a transformer merely because --model was supplied.
    if isinstance(loaded_model, TfidfIntentClassifier):
        tfidf_clf = loaded_model
        tfidf_name = "TF-IDF + Logistic Regression (Pre-trained)"
    else:
        if not train_ex:
            raise ValueError(
                "Cannot evaluate trainable baseline 'tfidf': No training split provided. "
                "Fitting trainable models on evaluation or test data is strictly prohibited."
            )
        tfidf_clf = TfidfIntentClassifier(seed=args.seed)
        tfidf_clf.fit(train_ex)
        tfidf_name = "TF-IDF + Logistic Regression (Fit on Train Split)"

    eval_results["models"]["tfidf_logistic_regression"] = {
        "name": tfidf_name,
        "metrics": _evaluate_classifier(tfidf_clf, test_ex, y_true),
    }

    if isinstance(loaded_model, EmbeddingIntentClassifier):
        transformer_clf = loaded_model
        transformer_name = f"Multilingual Transformer ({model_meta.get('model_identifier', PRETRAINED_MODEL_ID)}) + Logistic Regression (Pre-trained)"
    else:
        if not train_ex:
            raise ValueError(
                "Cannot evaluate trainable baseline 'transformer': No training split provided. "
                "Fitting trainable models on evaluation or test data is strictly prohibited."
            )
        transformer_clf = EmbeddingIntentClassifier(model_name=PRETRAINED_MODEL_ID, seed=args.seed)
        transformer_clf.fit(train_ex)
        transformer_name = f"Multilingual Transformer ({PRETRAINED_MODEL_ID}) + Logistic Regression (Fit on Train Split)"

    eval_results["models"]["transformer_logistic_regression"] = {
        "name": transformer_name,
        "metrics": _evaluate_classifier(transformer_clf, test_ex, y_true),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n=== INTENT EVALUATION REPORT ({data_path.name}) ===")
    print(f"Split Counts -> Train: {len(train_ex)} | Val: {len(val_ex)} | Test: {len(test_ex)}")
    print(f"Test Classes: {sorted(list(set(y_true)))}\n")
    print(f"{'Model':<65} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 103)
    for m_info in eval_results["models"].values():
        m_name = m_info["name"]
        metrics = m_info["metrics"]
        print(f"{m_name:<65} | {metrics.get('accuracy', 0.0):<10.4f} | {metrics.get('macro_f1', 0.0):<10.4f} | {metrics.get('weighted_f1', 0.0):<10.4f}")

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
