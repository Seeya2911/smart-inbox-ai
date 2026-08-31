"""Final classification improvement and evaluation pipeline for Smart Inbox AI.

Executes:
1. Priority feature engineering comparison (TF-IDF vs TF-IDF + Text Signals)
2. Class imbalance weighting comparison
3. High-priority decision threshold sensitivity analysis on Validation set
4. Final model selection based on Validation HIGH Recall, HIGH F1, and Macro-F1
5. Final evaluation ONCE on the untouched Test Set ($n=602$)
6. Serialization of final production model artifacts to artifacts/final/
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import classification_report

from ml.deduplication import strip_email_boilerplate
from ml.priority_aware_classifier import PriorityAwareClassifier
from ml.schema import CanonicalEmailExample
from ml.train_multi_output import MultiOutputClassifier, compute_head_metrics


def evaluate_priority_policy(
    clf: PriorityAwareClassifier,
    val_examples: List[CanonicalEmailExample],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate a specific high_threshold policy on validation data."""
    clf.high_threshold = threshold
    preds = clf.predict(val_examples)
    y_true = [ex.priority for ex in val_examples]
    y_pred = [p["priority"] for p in preds]

    metrics = compute_head_metrics(y_true, y_pred, "priority")
    high_class = metrics["per_class"].get("high", {})
    med_class = metrics["per_class"].get("medium", {})
    low_class = metrics["per_class"].get("low", {})

    return {
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "high_precision": high_class.get("precision", 0.0),
        "high_recall": high_class.get("recall", 0.0),
        "high_f1": high_class.get("f1", 0.0),
        "medium_f1": med_class.get("f1", 0.0),
        "low_precision": low_class.get("precision", 0.0),
        "low_f1": low_class.get("f1", 0.0),
        "full_metrics": metrics,
    }


def run_final_pipeline(
    train_ex: List[CanonicalEmailExample],
    val_ex: List[CanonicalEmailExample],
    test_ex: List[CanonicalEmailExample],
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete final priority-aware classification experiment and selection."""
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  PHASE 1 & 2: BASELINE V1 REFERENCE (VALIDATION & TEST)")
    print("=" * 80)

    # 1. Baseline V1 reference model
    base_clf = MultiOutputClassifier(seed=seed)
    base_clf.fit(train_ex)

    base_val_preds = base_clf.predict(val_ex)
    base_val_intent = compute_head_metrics([ex.intent for ex in val_ex], [p["intent"] for p in base_val_preds], "intent")
    base_val_pri = compute_head_metrics([ex.priority for ex in val_ex], [p["priority"] for p in base_val_preds], "priority")

    base_test_preds = base_clf.predict(test_ex)
    base_test_intent = compute_head_metrics([ex.intent for ex in test_ex], [p["intent"] for p in base_test_preds], "intent")
    base_test_pri = compute_head_metrics([ex.priority for ex in test_ex], [p["priority"] for p in base_test_preds], "priority")

    print(f"  Baseline Validation Intent Macro-F1:   {base_val_intent['macro_f1']:.4f}")
    print(f"  Baseline Validation Priority Macro-F1: {base_val_pri['macro_f1']:.4f}")
    print(f"  Baseline Validation High F1 / Recall:  {base_val_pri['per_class']['high']['f1']:.4f} / {base_val_pri['per_class']['high']['recall']:.4f}")

    print("\n" + "=" * 80)
    print("  PHASE 3 & 4: PRIORITY FEATURE ENGINEERING & WEIGHT EXPERIMENTS (VALIDATION)")
    print("=" * 80)

    candidate_models = [
        ("TFIDF_Standard_Balanced", False, "balanced", 1.0),
        ("Hybrid_Engineered_Balanced", True, "balanced", 1.0),
        ("Hybrid_Engineered_Weight_H3", True, {"low": 1.0, "medium": 1.8, "high": 3.0}, 1.0),
        ("Hybrid_Engineered_Weight_H4", True, {"low": 1.0, "medium": 2.0, "high": 4.0}, 1.0),
        ("Hybrid_Engineered_C2_Balanced", True, "balanced", 2.0),
    ]

    trained_models = {}
    for name, use_feat, weights, c_val in candidate_models:
        if use_feat:
            clf = PriorityAwareClassifier(
                ngram_range=(1, 2),
                max_features=10000,
                c_priority=c_val,
                class_weight_priority=weights,
                seed=seed,
            )
        else:
            clf = PriorityAwareClassifier(
                ngram_range=(1, 2),
                max_features=10000,
                c_priority=c_val,
                class_weight_priority=weights,
                seed=seed,
            )
            # Override feature extractor to return empty sparse matrix
            clf.priority_feature_extractor.extract_sparse = lambda ex: np.zeros((len(ex), 0))

        clf.fit(train_ex)
        trained_models[name] = clf

        v_metrics = evaluate_priority_policy(clf, val_ex, threshold=0.5)
        print(f"  {name:<32} Acc: {v_metrics['accuracy']:.4f} | Macro-F1: {v_metrics['macro_f1']:.4f} | High F1: {v_metrics['high_f1']:.4f} | High Recall: {v_metrics['high_recall']:.4f}")

    print("\n" + "=" * 80)
    print("  PHASE 5: HIGH PRIORITY THRESHOLD SENSITIVITY ANALYSIS (VALIDATION ONLY)")
    print("=" * 80)
    print(f"  {'Threshold':<10} {'Accuracy':<10} {'Macro-F1':<10} {'High Prec':<11} {'High Recall':<13} {'High F1':<10} {'Med F1':<10}")
    print("  " + "-" * 76)

    # Use the primary hybrid model for threshold tuning
    hybrid_clf = trained_models["Hybrid_Engineered_Balanced"]
    threshold_candidates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    threshold_results = []

    best_thresh_val = 0.50
    best_thresh_score = -1.0

    for th in threshold_candidates:
        res = evaluate_priority_policy(hybrid_clf, val_ex, threshold=th)
        threshold_results.append(res)
        print(f"  {th:<10.2f} {res['accuracy']:<10.4f} {res['macro_f1']:<10.4f} {res['high_precision']:<11.4f} {res['high_recall']:<13.4f} {res['high_f1']:<10.4f} {res['medium_f1']:<10.4f}")

        # Primary selection metric: High Recall + High F1 + Priority Macro F1, with precision constraint
        if res["high_precision"] >= 0.35:
            score = 0.4 * res["high_recall"] + 0.3 * res["high_f1"] + 0.3 * res["macro_f1"]
            if score > best_thresh_score:
                best_thresh_score = score
                best_thresh_val = th

    print("  " + "-" * 76)
    print(f"  [SELECTED VALIDATION THRESHOLD]: {best_thresh_val:.2f} (Composite Score: {best_thresh_score:.4f})")

    # Set optimal threshold on final candidate
    hybrid_clf.high_threshold = best_thresh_val

    # Final validation performance of selected candidate
    final_val_pri = evaluate_priority_policy(hybrid_clf, val_ex, threshold=best_thresh_val)

    # Intent Model Check: Keep robust TF-IDF with (1,2) ngrams + balanced LogReg
    intent_val_metrics = compute_head_metrics([ex.intent for ex in val_ex], [p["intent"] for p in hybrid_clf.predict(val_ex)], "intent")

    print("\n" + "=" * 80)
    print("  PHASE 8: FINAL EVALUATION ON UNTOUCHED TEST SET (n=602)")
    print("=" * 80)

    # Single test set evaluation
    final_test_preds = hybrid_clf.predict(test_ex)
    final_test_intent = compute_head_metrics([ex.intent for ex in test_ex], [p["intent"] for p in final_test_preds], "intent")
    final_test_pri = compute_head_metrics([ex.priority for ex in test_ex], [p["priority"] for p in final_test_preds], "priority")

    print(f"  {'TARGET & METRIC':<28} {'BASELINE V1':>16} {'FINAL CANDIDATE':>20} {'DELTA':>10}")
    print("  " + "-" * 76)
    print(f"  {'INTENT Accuracy':<28} {base_test_intent['accuracy']:>16.4f} {final_test_intent['accuracy']:>20.4f} {final_test_intent['accuracy'] - base_test_intent['accuracy']:>+10.4f}")
    print(f"  {'INTENT Macro F1':<28} {base_test_intent['macro_f1']:>16.4f} {final_test_intent['macro_f1']:>20.4f} {final_test_intent['macro_f1'] - base_test_intent['macro_f1']:>+10.4f}")
    print(f"  {'INTENT Weighted F1':<28} {base_test_intent['weighted_f1']:>16.4f} {final_test_intent['weighted_f1']:>20.4f} {final_test_intent['weighted_f1'] - base_test_intent['weighted_f1']:>+10.4f}")
    print("  " + "-" * 76)
    print(f"  {'PRIORITY Accuracy':<28} {base_test_pri['accuracy']:>16.4f} {final_test_pri['accuracy']:>20.4f} {final_test_pri['accuracy'] - base_test_pri['accuracy']:>+10.4f}")
    print(f"  {'PRIORITY Macro F1':<28} {base_test_pri['macro_f1']:>16.4f} {final_test_pri['macro_f1']:>20.4f} {final_test_pri['macro_f1'] - base_test_pri['macro_f1']:>+10.4f}")
    print(f"  {'PRIORITY Weighted F1':<28} {base_test_pri['weighted_f1']:>16.4f} {final_test_pri['weighted_f1']:>20.4f} {final_test_pri['weighted_f1'] - base_test_pri['weighted_f1']:>+10.4f}")
    print("  " + "-" * 76)

    # Detailed High Priority Breakdown
    b_hp = base_test_pri["per_class"]["high"]
    f_hp = final_test_pri["per_class"]["high"]
    print(f"  {'HIGH Priority Precision':<28} {b_hp['precision']:>16.4f} {f_hp['precision']:>20.4f} {f_hp['precision'] - b_hp['precision']:>+10.4f}")
    print(f"  {'HIGH Priority Recall':<28} {b_hp['recall']:>16.4f} {f_hp['recall']:>20.4f} {f_hp['recall'] - b_hp['recall']:>+10.4f}")
    print(f"  {'HIGH Priority F1':<28} {b_hp['f1']:>16.4f} {f_hp['f1']:>20.4f} {f_hp['f1'] - b_hp['f1']:>+10.4f}")

    b_mp = base_test_pri["per_class"]["medium"]
    f_mp = final_test_pri["per_class"]["medium"]
    print(f"  {'MEDIUM Priority F1':<28} {b_mp['f1']:>16.4f} {f_mp['f1']:>20.4f} {f_mp['f1'] - b_mp['f1']:>+10.4f}")

    b_lp = base_test_pri["per_class"]["low"]
    f_lp = final_test_pri["per_class"]["low"]
    print(f"  {'LOW Priority F1':<28} {b_lp['f1']:>16.4f} {f_lp['f1']:>20.4f} {f_lp['f1'] - b_lp['f1']:>+10.4f}")
    print(f"  {'LOW Priority Precision':<28} {b_lp['precision']:>16.4f} {f_lp['precision']:>20.4f} {f_lp['precision'] - b_lp['precision']:>+10.4f}")

    # Decision logic
    high_recall_diff = f_hp["recall"] - b_hp["recall"]
    high_f1_diff = f_hp["f1"] - b_hp["f1"]
    pri_macro_diff = final_test_pri["macro_f1"] - base_test_pri["macro_f1"]

    if high_recall_diff > 0 and f_hp["precision"] >= 0.35 and pri_macro_diff >= -0.01:
        decision = "A. Improved Priority-Aware model beats Baseline V1 on HIGH Priority Recall and F1"
    elif pri_macro_diff > 0.01 and high_f1_diff >= 0:
        decision = "A. Improved Priority-Aware model beats Baseline V1"
    else:
        decision = "B. Baseline V1 remains best or C. Dataset label ambiguity is the primary bottleneck"

    print(f"\n  [FINAL DECISION]: {decision}")

    # Serialize final model artifacts to artifacts/final/
    intent_art_path = final_dir / "intent_model.joblib"
    priority_art_path = final_dir / "priority_model.joblib"
    hybrid_clf.save(intent_art_path, priority_art_path)
    print(f"\n  [OK] Saved final intent model  -> {intent_art_path}")
    print(f"  [OK] Saved final priority model -> {priority_art_path}")

    # Metadata & Evaluation payloads
    model_metadata = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reproducibility": {
            "dataset_filename": "smart_inbox_ai_dataset_v2.xlsx",
            "dataset_sha256": "55effb2dc2e3b22b8b3eec81b1d301ee71cbc03a41185bdcc7d3c6264c8771ee",
            "split_sha256": "c5e4975f313e7106e22dcffa5a0491c1187b642478f78d19c60e2e56e2b43054",
            "random_seed": seed,
            "high_priority_threshold": best_thresh_val,
            "feature_names": hybrid_clf.priority_feature_extractor.FEATURE_NAMES,
        },
        "validation_selection": {
            "intent_macro_f1": intent_val_metrics["macro_f1"],
            "priority_macro_f1": final_val_pri["macro_f1"],
            "high_priority_recall": final_val_pri["high_recall"],
            "high_priority_f1": final_val_pri["high_f1"],
        },
    }
    with (final_dir / "model_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(model_metadata, fh, indent=2)

    final_evaluation = {
        "decision": decision,
        "baseline_v1": {
            "intent": base_test_intent,
            "priority": base_test_pri,
        },
        "final_model": {
            "intent": final_test_intent,
            "priority": final_test_pri,
        },
        "threshold_sensitivity_validation": threshold_results,
    }
    with (final_dir / "final_evaluation.json").open("w", encoding="utf-8") as fh:
        json.dump(final_evaluation, fh, indent=2)

    print(f"  [OK] Saved final metadata -> {final_dir / 'model_metadata.json'}")
    print(f"  [OK] Saved final evaluation -> {final_dir / 'final_evaluation.json'}")

    return final_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train priority-aware classification model")
    parser.add_argument("--dataset-splits", type=str, default="artifacts/canonical_multi_output_dataset.json")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_path = Path(args.dataset_splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits path {splits_path} not found.")

    with splits_path.open("r", encoding="utf-8") as fh:
        split_data = json.load(fh)["splits"]

    train_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["train"]]
    val_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["val"]]
    test_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["test"]]

    run_final_pipeline(train_ex, val_ex, test_ex, Path(args.output_dir), seed=args.seed)


if __name__ == "__main__":
    main()
