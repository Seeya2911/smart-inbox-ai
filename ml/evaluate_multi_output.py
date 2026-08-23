"""Multi-Output Evaluation Harness.

Evaluates multi-output classifiers on untouched Gold test data (gold/test.jsonl).
Produces accuracy, Macro F1, per-class Precision/Recall/F1, confusion matrices,
and performance breakdowns BROKEN OUT BY SOURCE (enron vs synthetic vs spam vs inbox).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import classification_report, confusion_matrix

from ml.schema import CanonicalEmailExample
from ml.train_multi_output import MultiOutputClassifier


def compute_head_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute accuracy, Macro F1, Weighted F1, per-class metrics, and confusion matrix."""
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(report.get("accuracy", 0.0)), 4),
        "macro_f1": round(float(report.get("macro avg", {}).get("f1-score", 0.0)), 4),
        "weighted_f1": round(float(report.get("weighted avg", {}).get("f1-score", 0.0)), 4),
        "per_class": {
            lbl: {
                "precision": round(float(metrics.get("precision", 0.0)), 4),
                "recall": round(float(metrics.get("recall", 0.0)), 4),
                "f1": round(float(metrics.get("f1-score", 0.0)), 4),
                "support": int(metrics.get("support", 0)),
            }
            for lbl, metrics in report.items()
            if lbl not in {"accuracy", "macro avg", "weighted avg"}
        },
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def evaluate_by_source(
    examples: List[CanonicalEmailExample],
    predictions: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Break down model accuracy and F1 metrics by dataset source."""
    sources = sorted(list({ex.source for ex in examples}))
    by_source_res: Dict[str, Any] = {}

    for src in sources:
        src_indices = [i for i, ex in enumerate(examples) if ex.source == src]
        src_examples = [examples[i] for i in src_indices]
        src_preds = [predictions[i] for i in src_indices]

        true_intents = [ex.intent for ex in src_examples]
        pred_intents = [p["intent"] for p in src_preds]

        true_priorities = [ex.priority for ex in src_examples]
        pred_priorities = [p["priority"] for p in src_preds]

        by_source_res[src] = {
            "example_count": len(src_examples),
            "intent_metrics": compute_head_metrics(true_intents, pred_intents),
            "priority_metrics": compute_head_metrics(true_priorities, pred_priorities),
        }

    return by_source_res


def run_evaluation(
    data_path: Path,
    model_path: Path,
    output_report_path: Path = Path("artifacts/multi_output_eval_results.json"),
) -> Dict[str, Any]:
    """Run full evaluation on gold test set and save report artifact."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset file not found: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact file not found: {model_path}")

    text_content = data_path.read_text(encoding="utf-8").strip()
    records = []
    if text_content.startswith("["):
        records = json.loads(text_content)
    else:
        lines = [l.strip() for l in text_content.splitlines() if l.strip()]
        records = [json.loads(l) for l in lines]

    test_examples = [CanonicalEmailExample.from_dict(r) for r in records]
    clf = MultiOutputClassifier.load(model_path)

    predictions = clf.predict(test_examples)

    true_intents = [ex.intent for ex in test_examples]
    pred_intents = [p["intent"] for p in predictions]

    true_priorities = [ex.priority for ex in test_examples]
    pred_priorities = [p["priority"] for p in predictions]

    overall_intent_metrics = compute_head_metrics(true_intents, pred_intents)
    overall_priority_metrics = compute_head_metrics(true_priorities, pred_priorities)
    source_breakdown = evaluate_by_source(test_examples, predictions)

    report = {
        "status": "success",
        "dataset_file": str(data_path),
        "model_file": str(model_path),
        "test_examples_count": len(test_examples),
        "overall_intent_metrics": overall_intent_metrics,
        "overall_priority_metrics": overall_priority_metrics,
        "performance_by_source": source_breakdown,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Output Evaluation Harness CLI")
    parser.add_argument("--data", type=str, default="gold/test.jsonl", help="Path to gold test split")
    parser.add_argument("--model", type=str, required=True, help="Path to multi-output model joblib artifact")
    parser.add_argument("--output", type=str, default="artifacts/multi_output_eval_results.json", help="Output JSON report path")
    args = parser.parse_args()

    report = run_evaluation(
        data_path=Path(args.data),
        model_path=Path(args.model),
        output_report_path=Path(args.output),
    )

    print(f"\n=== MULTI-OUTPUT EVALUATION REPORT ({Path(args.data).name}) ===")
    print(f"Total Test Examples: {report['test_examples_count']}")
    print(f"INTENT   -> Accuracy: {report['overall_intent_metrics']['accuracy']:.4f} | Macro F1: {report['overall_intent_metrics']['macro_f1']:.4f}")
    print(f"PRIORITY -> Accuracy: {report['overall_priority_metrics']['accuracy']:.4f} | Macro F1: {report['overall_priority_metrics']['macro_f1']:.4f}")

    print("\n--- Performance by Source ---")
    for src, metrics in report["performance_by_source"].items():
        print(f"Source: {src:<12} (N={metrics['example_count']:<3}) | Intent Macro F1: {metrics['intent_metrics']['macro_f1']:.4f} | Priority Macro F1: {metrics['priority_metrics']['macro_f1']:.4f}")

    print(f"\nFull evaluation report saved to: {args.output}")


if __name__ == "__main__":
    main()
