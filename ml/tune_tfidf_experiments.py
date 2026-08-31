"""Controlled hyperparameter tuning and model improvement experiments for TF-IDF + LogisticRegression.

Evaluates:
- Experiment A: TF-IDF feature representations (ngram_range, min_df, sublinear_tf, max_features)
- Experiment B: Regularization strength C (0.1, 0.5, 1.0, 2.0, 5.0)

DISCIPLINE:
- Trained on Train split (2,102)
- Evaluated and selected on Validation split (637)
- Primary metric: Validation Macro-F1 (and tracking High priority F1/Recall)
- Best model evaluated ONCE on untouched Test split (602)
- Baseline V1 artifacts are preserved without modification.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.deduplication import strip_email_boilerplate
from ml.schema import CanonicalEmailExample
from ml.train_multi_output import MultiOutputClassifier, compute_head_metrics


# ---------------------------------------------------------------------------
# TunableMultiOutputClassifier
# ---------------------------------------------------------------------------

class TunableMultiOutputClassifier:
    """Configurable multi-output classifier for hyperparameter grid search."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
        max_features: int = 10000,
        c_intent: float = 1.0,
        c_priority: float = 1.0,
        class_weight: str = "balanced",
        seed: int = 42,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features
        self.c_intent = c_intent
        self.c_priority = c_priority
        self.class_weight = class_weight
        self.seed = seed

        self.intent_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
        )
        self.priority_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
        )

        self.intent_head = LogisticRegression(
            C=c_intent,
            max_iter=1000,
            class_weight=class_weight,
            random_state=seed,
        )
        self.priority_head = LogisticRegression(
            C=c_priority,
            max_iter=1000,
            class_weight=class_weight,
            random_state=seed,
        )
        self.is_fitted = False

    def _prepare_texts(self, examples: List[CanonicalEmailExample]) -> List[str]:
        return [strip_email_boilerplate(ex.full_text) for ex in examples]

    def fit(self, train_examples: List[CanonicalEmailExample]) -> "TunableMultiOutputClassifier":
        texts = self._prepare_texts(train_examples)
        y_intent = [ex.intent for ex in train_examples]
        y_priority = [ex.priority for ex in train_examples]

        X_intent = self.intent_vectorizer.fit_transform(texts)
        X_priority = self.priority_vectorizer.fit_transform(texts)

        self.intent_head.fit(X_intent, y_intent)
        self.priority_head.fit(X_priority, y_priority)
        self.is_fitted = True
        return self

    def predict(self, examples: List[CanonicalEmailExample]) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict.")
        if not examples:
            return []

        texts = self._prepare_texts(examples)
        X_intent = self.intent_vectorizer.transform(texts)
        X_priority = self.priority_vectorizer.transform(texts)

        preds_i = self.intent_head.predict(X_intent)
        preds_p = self.priority_head.predict(X_priority)
        proba_i = self.intent_head.predict_proba(X_intent)
        proba_p = self.priority_head.predict_proba(X_priority)

        return [
            {
                "intent": str(pi),
                "priority": str(pp),
                "intent_confidence": float(np.max(pbi)),
                "priority_confidence": float(np.max(pbp)),
            }
            for pi, pp, pbi, pbp in zip(preds_i, preds_p, proba_i, proba_p)
        ]

    def save(self, intent_path: Path, priority_path: Path) -> None:
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_identifier": "tuned-tfidf-intent-logistic-regression",
                "vectorizer": self.intent_vectorizer,
                "classifier": self.intent_head,
                "config": {
                    "ngram_range": self.ngram_range,
                    "min_df": self.min_df,
                    "sublinear_tf": self.sublinear_tf,
                    "max_features": self.max_features,
                    "C": self.c_intent,
                },
            },
            intent_path,
        )
        joblib.dump(
            {
                "model_identifier": "tuned-tfidf-priority-logistic-regression",
                "vectorizer": self.priority_vectorizer,
                "classifier": self.priority_head,
                "config": {
                    "ngram_range": self.ngram_range,
                    "min_df": self.min_df,
                    "sublinear_tf": self.sublinear_tf,
                    "max_features": self.max_features,
                    "C": self.c_priority,
                },
            },
            priority_path,
        )


# ---------------------------------------------------------------------------
# Grid Search Runner
# ---------------------------------------------------------------------------

def run_tuning_grid(
    train_ex: List[CanonicalEmailExample],
    val_ex: List[CanonicalEmailExample],
    test_ex: List[CanonicalEmailExample],
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute grid search across controlled configurations, evaluate on Validation, test the best."""
    exp_dir = output_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline configuration as reference
    baseline_clf = TunableMultiOutputClassifier(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=10000, c_intent=1.0, c_priority=1.0, seed=seed
    )
    baseline_clf.fit(train_ex)
    base_val_preds = baseline_clf.predict(val_ex)
    base_v_intent = compute_head_metrics([ex.intent for ex in val_ex], [p["intent"] for p in base_val_preds], "intent")
    base_v_pri = compute_head_metrics([ex.priority for ex in val_ex], [p["priority"] for p in base_val_preds], "priority")

    # Define candidate configurations
    candidate_configs = [
        # (name, ngram_range, min_df, sublinear_tf, max_features, C)
        ("Config_1_Baseline", (1, 2), 1, True, 10000, 1.0),
        ("Config_2_Unigrams", (1, 1), 1, True, 10000, 1.0),
        ("Config_3_Trigrams", (1, 3), 1, True, 15000, 1.0),
        ("Config_4_Trigrams_20k", (1, 3), 2, True, 20000, 1.0),
        ("Config_5_MinDF2", (1, 2), 2, True, 10000, 1.0),
        ("Config_6_NoSublinear", (1, 2), 1, False, 10000, 1.0),
        ("Config_7_C_0.5", (1, 2), 1, True, 10000, 0.5),
        ("Config_8_C_2.0", (1, 2), 1, True, 10000, 2.0),
        ("Config_9_C_5.0", (1, 2), 1, True, 10000, 5.0),
        ("Config_10_Trigrams_C2", (1, 3), 1, True, 15000, 2.0),
        ("Config_11_Trigrams_C5", (1, 3), 1, True, 20000, 5.0),
    ]

    results = []
    best_config = None
    best_combined_score = -1.0

    print("\n" + "=" * 80)
    print("  CONTROLLED TF-IDF HYPERPARAMETER EXPERIMENTS (VALIDATION EVALUATION)")
    print("=" * 80)
    print(f"  {'Config Name':<24} {'Intent Macro-F1':>16} {'Priority Macro-F1':>18} {'High-Priority F1':>18}")
    print("  " + "-" * 78)

    for name, ngram, min_df, sublinear, max_feat, c_val in candidate_configs:
        clf = TunableMultiOutputClassifier(
            ngram_range=ngram,
            min_df=min_df,
            sublinear_tf=sublinear,
            max_features=max_feat,
            c_intent=c_val,
            c_priority=c_val,
            seed=seed,
        )
        clf.fit(train_ex)
        v_preds = clf.predict(val_ex)

        v_intent = compute_head_metrics([ex.intent for ex in val_ex], [p["intent"] for p in v_preds], "intent")
        v_pri = compute_head_metrics([ex.priority for ex in val_ex], [p["priority"] for p in v_preds], "priority")

        i_f1 = v_intent["macro_f1"]
        p_f1 = v_pri["macro_f1"]
        high_f1 = v_pri["per_class"].get("high", {}).get("f1", 0.0)
        high_recall = v_pri["per_class"].get("high", {}).get("recall", 0.0)
        high_prec = v_pri["per_class"].get("high", {}).get("precision", 0.0)

        # Combined metric for selection
        combined_score = 0.5 * i_f1 + 0.5 * p_f1

        record = {
            "name": name,
            "params": {
                "ngram_range": list(ngram),
                "min_df": min_df,
                "sublinear_tf": sublinear,
                "max_features": max_feat,
                "C": c_val,
            },
            "validation_metrics": {
                "intent_accuracy": v_intent["accuracy"],
                "intent_macro_f1": i_f1,
                "intent_weighted_f1": v_intent["weighted_f1"],
                "priority_accuracy": v_pri["accuracy"],
                "priority_macro_f1": p_f1,
                "priority_weighted_f1": v_pri["weighted_f1"],
                "high_priority_f1": high_f1,
                "high_priority_recall": high_recall,
                "high_priority_precision": high_prec,
                "combined_macro_f1": round(combined_score, 4),
            },
        }
        results.append(record)

        print(f"  {name:<24} {i_f1:>16.4f} {p_f1:>18.4f} {high_f1:>18.4f}")

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_config = (name, clf, record)

    assert best_config is not None
    best_name, best_clf, best_record = best_config
    print("  " + "-" * 78)
    print(f"  [WINNER ON VALIDATION]: {best_name} (Combined Macro-F1 = {best_combined_score:.4f})")

    # 2. Evaluate Best Config ONCE on Untouched Test Set
    print("\n" + "=" * 80)
    print("  FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 80)

    # Baseline test metrics for comparison
    base_t_preds = baseline_clf.predict(test_ex)
    base_t_intent = compute_head_metrics([ex.intent for ex in test_ex], [p["intent"] for p in base_t_preds], "intent")
    base_t_pri = compute_head_metrics([ex.priority for ex in test_ex], [p["priority"] for p in base_t_preds], "priority")

    # Best model test metrics
    best_t_preds = best_clf.predict(test_ex)
    best_t_intent = compute_head_metrics([ex.intent for ex in test_ex], [p["intent"] for p in best_t_preds], "intent")
    best_t_pri = compute_head_metrics([ex.priority for ex in test_ex], [p["priority"] for p in best_t_preds], "priority")

    print(f"  {'METRIC':<26} {'BASELINE V1':>16} {'BEST TUNED (' + best_name + ')':>28} {'DELTA':>10}")
    print("  " + "-" * 82)
    for target, b_m, t_m in [("INTENT", base_t_intent, best_t_intent), ("PRIORITY", base_t_pri, best_t_pri)]:
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            b_val = b_m[metric]
            t_val = t_m[metric]
            delta = t_val - b_val
            sign = "+" if delta >= 0 else ""
            print(f"  {f'{target} {metric}':<26} {b_val:>16.4f} {t_val:>28.4f} {f'{sign}{delta:.4f}':>10}")
        print("  " + "-" * 82)

    # Compare High Priority
    b_h_f1 = base_t_pri["per_class"].get("high", {}).get("f1", 0.0)
    t_h_f1 = best_t_pri["per_class"].get("high", {}).get("f1", 0.0)
    b_h_rec = base_t_pri["per_class"].get("high", {}).get("recall", 0.0)
    t_h_rec = best_t_pri["per_class"].get("high", {}).get("recall", 0.0)
    print(f"  HIGH-PRIORITY F1:        Baseline={b_h_f1:.4f}, Best Tuned={t_h_f1:.4f} (Delta: {t_h_f1 - b_h_f1:+.4f})")
    print(f"  HIGH-PRIORITY Recall:    Baseline={b_h_rec:.4f}, Best Tuned={t_h_rec:.4f} (Delta: {t_h_rec - b_h_rec:+.4f})")

    # Determine Decision: A / B / C
    intent_diff = best_t_intent["macro_f1"] - base_t_intent["macro_f1"]
    priority_diff = best_t_pri["macro_f1"] - base_t_pri["macro_f1"]

    if intent_diff > 0.01 and priority_diff > 0.01:
        decision = "A. Improved TF-IDF model clearly beats Baseline V1"
    elif abs(intent_diff) <= 0.01 and abs(priority_diff) <= 0.01:
        decision = "B. Baseline V1 remains best (tuning improvements are negligible or within noise)"
    else:
        decision = "C. Evidence indicates main bottleneck is label/dataset ambiguity rather than model parameters"

    # Save artifacts into experiments dir
    intent_art_path = exp_dir / "intent_tfidf_tuned.joblib"
    priority_art_path = exp_dir / "priority_tfidf_tuned.joblib"
    best_clf.save(intent_art_path, priority_art_path)
    print(f"\n  [OK] Saved tuned artifacts -> {intent_art_path}, {priority_art_path}")

    comparison_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiments_summary": results,
        "selected_best_configuration": {
            "name": best_name,
            "params": best_record["params"],
            "validation_metrics": best_record["validation_metrics"],
        },
        "test_comparison": {
            "baseline_v1": {
                "intent": base_t_intent,
                "priority": base_t_pri,
            },
            "best_tuned": {
                "intent": best_t_intent,
                "priority": best_t_pri,
            },
            "delta": {
                "intent_macro_f1": round(intent_diff, 4),
                "priority_macro_f1": round(priority_diff, 4),
                "high_priority_f1": round(t_h_f1 - b_h_f1, 4),
            },
        },
        "decision": decision,
    }

    comp_path = exp_dir / "classification_comparison.json"
    with comp_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison_payload, fh, indent=2)
    print(f"  [OK] Saved experiment comparison -> {comp_path}")

    return comparison_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF hyperparameter tuning grid")
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

    run_tuning_grid(train_ex, val_ex, test_ex, Path(args.output_dir), seed=args.seed)


if __name__ == "__main__":
    main()
