"""Multi-Output Baseline Classifier Trainer — Intent + Priority (TF-IDF + Logistic Regression).

This is the primary supervised training pipeline for Smart Inbox AI v2.

Two independent classifiers are trained:
    Model 1 — Intent:   TF-IDF(ngram=(1,2), max=10000) + LogReg(balanced, seed=42)
    Model 2 — Priority: TF-IDF(ngram=(1,2), max=10000) + LogReg(balanced, seed=42)

Each model has its own dedicated TF-IDF vectorizer.

LEAKAGE SAFETY
--------------
Model inputs: subject + body (via CanonicalEmailExample.full_text)
Model inputs do NOT include: intent, priority, intent_reason, priority_reason,
label_confidence, source, label_source, is_synthetic, id/index.

TRAINING CONFIGURATION (explicit, locked)
------------------------------------------
TfidfVectorizer:
    ngram_range=(1, 2)
    max_features=10000
    sublinear_tf=True

LogisticRegression:
    max_iter=1000
    class_weight="balanced"
    random_state=42
    C=1.0

Note: LogisticRegression is NOT epoch-based. max_iter controls the
optimization solver iteration limit. There is no "epochs" concept here.

PREDICTION OUTPUT
-----------------
    {
        "intent": "...",
        "priority": "...",
        "intent_confidence": 0.0–1.0,   # from predict_proba()
        "priority_confidence": 0.0–1.0, # from predict_proba()
    }

CLI USAGE
---------
    python -m ml.train_multi_output \\
        --data smart_inbox_ai_dataset_v2.xlsx \\
        --output-dir artifacts/ \\
        --seed 42

    python -m ml.train_multi_output \\
        --data data/smart_inbox_ai_dataset_v2.jsonl \\
        --output-dir artifacts/ \\
        --seed 42
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ml.deduplication import strip_email_boilerplate
from ml.schema import ALLOWED_INTENTS, ALLOWED_PRIORITIES, CanonicalEmailExample


# ---------------------------------------------------------------------------
# Locked hyperparameters — do NOT change without updating metadata/experiments
# ---------------------------------------------------------------------------
_TFIDF_NGRAM_RANGE = (1, 2)
_TFIDF_MAX_FEATURES = 10000
_LR_MAX_ITER = 1000
_LR_C = 1.0
_LR_CLASS_WEIGHT = "balanced"


# ---------------------------------------------------------------------------
# MultiOutputClassifier
# ---------------------------------------------------------------------------

class MultiOutputClassifier:
    """TF-IDF + Logistic Regression classifier with independent Intent and Priority heads.

    Each head has its own dedicated TF-IDF vectorizer. They share nothing except
    the input text (subject + body from CanonicalEmailExample.full_text).

    LEAKAGE NOTE: _prepare_texts() extracts only subject + body. Metadata
    fields (intent, priority, intent_reason, priority_reason, label_confidence,
    source, is_synthetic) are never passed to either vectorizer.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = _TFIDF_NGRAM_RANGE,
        max_features: int = _TFIDF_MAX_FEATURES,
        lr_max_iter: int = _LR_MAX_ITER,
        lr_c: float = _LR_C,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.lr_max_iter = lr_max_iter
        self.lr_c = lr_c

        # Two independent vectorizers — one per head
        self.intent_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.priority_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )

        self.intent_head = LogisticRegression(
            C=lr_c,
            max_iter=lr_max_iter,
            class_weight=_LR_CLASS_WEIGHT,
            random_state=seed,
        )
        self.priority_head = LogisticRegression(
            C=lr_c,
            max_iter=lr_max_iter,
            class_weight=_LR_CLASS_WEIGHT,
            random_state=seed,
        )
        self.is_fitted = False

    def _prepare_texts(self, examples: List[CanonicalEmailExample]) -> List[str]:
        """Extract model input text from examples.

        Uses subject + body via CanonicalEmailExample.full_text, then strips
        email boilerplate.  NEVER includes intent, priority, or any metadata.
        """
        return [strip_email_boilerplate(ex.full_text) for ex in examples]

    def fit(self, train_examples: List[CanonicalEmailExample]) -> "MultiOutputClassifier":
        """Fit both independent heads on training examples only."""
        if not train_examples:
            raise ValueError("Cannot fit classifier on empty training set.")

        texts = self._prepare_texts(train_examples)
        intent_targets = [ex.intent for ex in train_examples]
        priority_targets = [ex.priority for ex in train_examples]

        # Intent head: independent vectorizer + classifier
        X_intent = self.intent_vectorizer.fit_transform(texts)
        self.intent_head.fit(X_intent, intent_targets)

        # Priority head: independent vectorizer + classifier
        X_priority = self.priority_vectorizer.fit_transform(texts)
        self.priority_head.fit(X_priority, priority_targets)

        self.is_fitted = True
        return self

    def predict(self, examples: List[CanonicalEmailExample]) -> List[Dict[str, Any]]:
        """Predict intent and priority with confidence scores.

        Returns:
            List of dicts:
            {
                "intent": str,
                "priority": str,
                "intent_confidence": float,   # max prob from predict_proba()
                "priority_confidence": float, # max prob from predict_proba()
            }
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fit before calling predict.")
        if not examples:
            return []

        texts = self._prepare_texts(examples)

        X_intent = self.intent_vectorizer.transform(texts)
        X_priority = self.priority_vectorizer.transform(texts)

        intent_preds = self.intent_head.predict(X_intent)
        priority_preds = self.priority_head.predict(X_priority)

        intent_probas = self.intent_head.predict_proba(X_intent)
        priority_probas = self.priority_head.predict_proba(X_priority)

        return [
            {
                "intent": str(ip),
                "priority": str(pp),
                "intent_confidence": float(np.max(iprob)),
                "priority_confidence": float(np.max(pprob)),
            }
            for ip, pp, iprob, pprob in zip(
                intent_preds, priority_preds, intent_probas, priority_probas
            )
        ]

    def predict_proba_full(
        self, examples: List[CanonicalEmailExample]
    ) -> List[Dict[str, Any]]:
        """Return full probability distributions over all classes."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fit before calling predict_proba_full.")
        texts = self._prepare_texts(examples)
        X_intent = self.intent_vectorizer.transform(texts)
        X_priority = self.priority_vectorizer.transform(texts)
        intent_probas = self.intent_head.predict_proba(X_intent)
        priority_probas = self.priority_head.predict_proba(X_priority)
        intent_classes = list(self.intent_head.classes_)
        priority_classes = list(self.priority_head.classes_)
        return [
            {
                "intent_proba": dict(zip(intent_classes, [float(p) for p in irow])),
                "priority_proba": dict(zip(priority_classes, [float(p) for p in prow])),
            }
            for irow, prow in zip(intent_probas, priority_probas)
        ]

    def save_intent(self, filepath: Path) -> None:
        """Save intent head (vectorizer + classifier) as a joblib artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_identifier": "tfidf-intent-logistic-regression",
                "vectorizer": self.intent_vectorizer,
                "classifier": self.intent_head,
                "classes": list(self.intent_head.classes_),
                "seed": self.seed,
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
            },
            filepath,
        )

    def save_priority(self, filepath: Path) -> None:
        """Save priority head (vectorizer + classifier) as a joblib artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_identifier": "tfidf-priority-logistic-regression",
                "vectorizer": self.priority_vectorizer,
                "classifier": self.priority_head,
                "classes": list(self.priority_head.classes_),
                "seed": self.seed,
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
            },
            filepath,
        )

    @classmethod
    def load(cls, intent_path: Path, priority_path: Path) -> "MultiOutputClassifier":
        """Load a fitted classifier from two separate artifact files."""
        intent_artifact = joblib.load(intent_path)
        priority_artifact = joblib.load(priority_path)
        seed = intent_artifact.get("seed", 42)
        clf = cls(seed=seed)
        clf.intent_vectorizer = intent_artifact["vectorizer"]
        clf.intent_head = intent_artifact["classifier"]
        clf.priority_vectorizer = priority_artifact["vectorizer"]
        clf.priority_head = priority_artifact["classifier"]
        clf.is_fitted = True
        return clf

    # Legacy single-file load for backward compatibility with existing tests
    @classmethod
    def load_legacy(cls, filepath: Path) -> "MultiOutputClassifier":
        """Load from the old single-file format (vectorizer + intent_head + priority_head)."""
        artifact = joblib.load(filepath)
        seed = artifact.get("seed", 42)
        clf = cls(seed=seed)
        clf.intent_vectorizer = artifact["vectorizer"]
        clf.intent_head = artifact["intent_head"]
        clf.priority_vectorizer = artifact.get("priority_vectorizer", artifact["vectorizer"])
        clf.priority_head = artifact["priority_head"]
        clf.is_fitted = True
        return clf


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_head_metrics(y_true: List[str], y_pred: List[str], label_name: str) -> Dict[str, Any]:
    """Compute full per-class metrics and confusion matrix for one head."""
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    acc = float(accuracy_score(y_true, y_pred))

    # Majority baseline: always predict most common class
    from collections import Counter
    most_common = Counter(y_true).most_common(1)[0][0]
    baseline_preds = [most_common] * len(y_true)
    baseline_acc = float(accuracy_score(y_true, baseline_preds))
    baseline_macro_f1 = float(f1_score(y_true, baseline_preds, average="macro", zero_division=0))

    return {
        "label": label_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
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
        "majority_baseline": {
            "class": most_common,
            "accuracy": round(baseline_acc, 4),
            "macro_f1": round(baseline_macro_f1, 4),
        },
    }


def _print_head_metrics(metrics: Dict[str, Any]) -> None:
    """Print formatted evaluation report for one head."""
    label = metrics.get("label", "?")
    sep = "-" * 60
    print(f"\n  === {label.upper()} MODEL EVALUATION ===")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro  F1  : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    maj = metrics.get("majority_baseline", {})
    print(
        f"  Majority baseline: class='{maj.get('class')}' "
        f"acc={maj.get('accuracy',0):.4f} macro_f1={maj.get('macro_f1',0):.4f}"
    )
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<22} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print(f"  {sep}")
    for cls, vals in sorted(metrics.get("per_class", {}).items()):
        print(
            f"  {cls:<22} {vals['precision']:>6.3f} {vals['recall']:>6.3f} "
            f"{vals['f1']:>6.3f} {vals['support']:>6}"
        )

    cm_info = metrics.get("confusion_matrix", {})
    if cm_info.get("labels") and cm_info.get("matrix"):
        print(f"\n  Confusion matrix:")
        lbls = cm_info["labels"]
        # Header row
        hdr = "  " + " " * 22 + " ".join(f"{l[:6]:>7}" for l in lbls)
        print(hdr)
        for r_lbl, row in zip(lbls, cm_info["matrix"]):
            row_str = "  " + f"{r_lbl:<22}" + " ".join(f"{v:>7}" for v in row)
            print(row_str)


def _compute_intent_priority_crosstab(
    examples: List[CanonicalEmailExample],
) -> Dict[str, Any]:
    """Compute P(priority | intent) and raw counts for intent×priority analysis."""
    from collections import Counter
    joint: Counter[str] = Counter(f"{ex.intent}|{ex.priority}" for ex in examples)
    intent_totals: Counter[str] = Counter(ex.intent for ex in examples)

    all_intents = sorted(ALLOWED_INTENTS)
    all_priorities = sorted(ALLOWED_PRIORITIES)

    counts: Dict[str, Dict[str, int]] = {}
    cond_prob: Dict[str, Dict[str, float]] = {}
    for intent in all_intents:
        counts[intent] = {}
        cond_prob[intent] = {}
        total = intent_totals.get(intent, 0)
        for priority in all_priorities:
            cnt = joint.get(f"{intent}|{priority}", 0)
            counts[intent][priority] = cnt
            cond_prob[intent][priority] = round(cnt / total, 4) if total > 0 else 0.0

    return {
        "counts": counts,
        "conditional_priority_given_intent": cond_prob,
        "zero_count_combinations": [
            f"{i}|{p}"
            for i in all_intents
            for p in all_priorities
            if counts[i].get(p, 0) == 0
        ],
    }


def _get_env_metadata() -> Dict[str, str]:
    """Capture environment versions for reproducibility."""
    import sklearn
    meta = {
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import sentence_transformers
        meta["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        meta["sentence_transformers"] = "not-installed"
    return meta


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Source / synthetic breakdown
# ---------------------------------------------------------------------------

def _evaluate_by_source(
    examples: List[CanonicalEmailExample],
    preds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sources = sorted({ex.source for ex in examples})
    result: Dict[str, Any] = {}
    for src in sources:
        indices = [i for i, ex in enumerate(examples) if ex.source == src]
        src_examples = [examples[i] for i in indices]
        src_preds = [preds[i] for i in indices]
        result[src] = {
            "count": len(src_examples),
            "intent": compute_head_metrics(
                [ex.intent for ex in src_examples],
                [p["intent"] for p in src_preds],
                "intent",
            ),
            "priority": compute_head_metrics(
                [ex.priority for ex in src_examples],
                [p["priority"] for p in src_preds],
                "priority",
            ),
        }
    return result


def _evaluate_by_synthetic(
    examples: List[CanonicalEmailExample],
    preds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    groups = {"real": [], "synthetic": []}
    for i, ex in enumerate(examples):
        key = "synthetic" if ex.is_synthetic else "real"
        groups[key].append(i)
    result: Dict[str, Any] = {}
    for group_name, indices in groups.items():
        if not indices:
            result[group_name] = {"count": 0}
            continue
        gexs = [examples[i] for i in indices]
        gpreds = [preds[i] for i in indices]
        result[group_name] = {
            "count": len(gexs),
            "intent_macro_f1": round(
                float(f1_score(
                    [ex.intent for ex in gexs],
                    [p["intent"] for p in gpreds],
                    average="macro", zero_division=0,
                )), 4
            ),
            "priority_macro_f1": round(
                float(f1_score(
                    [ex.priority for ex in gexs],
                    [p["priority"] for p in gpreds],
                    average="macro", zero_division=0,
                )), 4
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    examples: List[CanonicalEmailExample],
    output_dir: Path,
    data_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full training pipeline: data quality → split → train → evaluate → save.

    Parameters
    ----------
    examples:
        Pre-loaded CanonicalEmailExample list from v2_dataset_loader.
    output_dir:
        Directory for all artifacts.
    data_path:
        Original data file path (for metadata/checksum).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Full experiment report dict.
    """
    from ml.data_quality import DataQualityError, check_email_dataset_integrity
    from ml.dataset_splitter import split_email_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62)
    print("  STEP 1: Dataset Integrity Check")
    print("=" * 62)
    quality_stats = check_email_dataset_integrity(examples, allow_internal_duplicates=False)
    print(f"  [OK] Passed -- {quality_stats['total_checked']} examples, "
          f"{quality_stats['unique_ids']} unique IDs, "
          f"{quality_stats['unique_normalized_texts']} unique texts")

    print("\n" + "=" * 62)
    print("  STEP 2: Deterministic Stratified Splitting (70/15/15)")
    print("=" * 62)
    t0 = time.time()
    train_ex, val_ex, test_ex = split_email_dataset(
        examples,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        print_distributions=True,
    )
    split_time = time.time() - t0
    print(f"  [OK] Split done in {split_time:.1f}s -- "
          f"train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

    print("\n" + "=" * 62)
    print("  STEP 3: Training Baseline (TF-IDF + Logistic Regression)")
    print("=" * 62)
    print(f"  Config: TfidfVectorizer(ngram_range={_TFIDF_NGRAM_RANGE}, max_features={_TFIDF_MAX_FEATURES})")
    print(f"  Config: LogisticRegression(max_iter={_LR_MAX_ITER}, C={_LR_C}, class_weight='balanced', seed={seed})")
    print("  Note: Two independent vectorizers -- one per head (intent / priority)")

    clf = MultiOutputClassifier(seed=seed)
    t_fit = time.time()
    clf.fit(train_ex)
    fit_time = time.time() - t_fit
    print(f"  [OK] Training complete in {fit_time:.2f}s")

    print("\n" + "=" * 62)
    print("  STEP 4: Validation Evaluation (for model selection -- test NOT touched)")
    print("=" * 62)
    t_val = time.time()
    val_preds = clf.predict(val_ex)
    val_intent_metrics = compute_head_metrics(
        [ex.intent for ex in val_ex], [p["intent"] for p in val_preds], "intent"
    )
    val_priority_metrics = compute_head_metrics(
        [ex.priority for ex in val_ex], [p["priority"] for p in val_preds], "priority"
    )
    val_time = time.time() - t_val
    print(f"  Validation set (n={len(val_ex)}):")
    _print_head_metrics(val_intent_metrics)
    _print_head_metrics(val_priority_metrics)

    print("\n" + "=" * 62)
    print("  STEP 5: Final Test Evaluation (untouched test set)")
    print("=" * 62)
    t_test = time.time()
    test_preds = clf.predict(test_ex)
    test_intent_metrics = compute_head_metrics(
        [ex.intent for ex in test_ex], [p["intent"] for p in test_preds], "intent"
    )
    test_priority_metrics = compute_head_metrics(
        [ex.priority for ex in test_ex], [p["priority"] for p in test_preds], "priority"
    )
    test_time = time.time() - t_test
    print(f"  Test set (n={len(test_ex)}):")
    _print_head_metrics(test_intent_metrics)
    _print_head_metrics(test_priority_metrics)

    print("\n" + "=" * 62)
    print("  STEP 6: Intent x Priority Analysis (test set)")
    print("=" * 62)
    crosstab = _compute_intent_priority_crosstab(test_ex)
    zero_combos = crosstab.get("zero_count_combinations", [])
    print(f"  Zero-count combinations in test set: {len(zero_combos)}")
    if zero_combos:
        print(f"    {zero_combos}")

    print("\n" + "=" * 62)
    print("  STEP 7: Source / Synthetic Analysis (test set)")
    print("=" * 62)
    source_breakdown = _evaluate_by_source(test_ex, test_preds)
    synthetic_breakdown = _evaluate_by_synthetic(test_ex, test_preds)
    for src, metrics in source_breakdown.items():
        print(f"  Source '{src}' (n={metrics['count']}): "
              f"intent macro-F1={metrics['intent']['macro_f1']:.4f}, "
              f"priority macro-F1={metrics['priority']['macro_f1']:.4f}")
    for grp, metrics in synthetic_breakdown.items():
        if metrics.get("count", 0) == 0:
            continue
        print(f"  {grp} (n={metrics['count']}): "
              f"intent macro-F1={metrics.get('intent_macro_f1', 0):.4f}, "
              f"priority macro-F1={metrics.get('priority_macro_f1', 0):.4f}")

    print("\n" + "=" * 62)
    print("  STEP 8: Saving Artifacts")
    print("=" * 62)

    intent_model_path = output_dir / "intent_tfidf.joblib"
    priority_model_path = output_dir / "priority_tfidf.joblib"
    clf.save_intent(intent_model_path)
    clf.save_priority(priority_model_path)
    print(f"  [OK] Intent model  -> {intent_model_path}")
    print(f"  [OK] Priority model -> {priority_model_path}")

    # Save canonical dataset with splits
    ds_path = output_dir / "canonical_multi_output_dataset.json"
    with ds_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "splits": {
                    "train": [ex.to_dict() for ex in train_ex],
                    "val": [ex.to_dict() for ex in val_ex],
                    "test": [ex.to_dict() for ex in test_ex],
                }
            },
            fh, indent=2, ensure_ascii=False,
        )
    print(f"  [OK] Dataset splits -> {ds_path}")

    # Build full metadata
    ds_checksum = _file_sha256(data_path) if data_path else "not-available"
    metadata: Dict[str, Any] = {
        "experiment": "tfidf-multi-output-baseline",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": seed,
        "environment": _get_env_metadata(),
        "dataset": {
            "path": str(data_path) if data_path else "in-memory",
            "sha256": ds_checksum,
            "total_examples": len(examples),
        },
        "split_config": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "strategy": "joint_intent_priority_with_intent_fallback",
        },
        "split_counts": {
            "train": len(train_ex),
            "val": len(val_ex),
            "test": len(test_ex),
        },
        "training_config": {
            "model": "TfidfVectorizer + LogisticRegression",
            "two_independent_vectorizers": True,
            "ngram_range": list(_TFIDF_NGRAM_RANGE),
            "max_features": _TFIDF_MAX_FEATURES,
            "sublinear_tf": True,
            "lr_max_iter": _LR_MAX_ITER,
            "lr_C": _LR_C,
            "lr_class_weight": _LR_CLASS_WEIGHT,
            "note": "LogisticRegression is not epoch-based; max_iter controls solver iterations.",
        },
        "artifacts": {
            "intent_model": str(intent_model_path),
            "priority_model": str(priority_model_path),
            "dataset": str(ds_path),
        },
    }
    metadata_path = output_dir / "multi_output_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"  [OK] Metadata -> {metadata_path}")

    # Save evaluation reports
    intent_eval_path = output_dir / "intent_evaluation.json"
    with intent_eval_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"validation": val_intent_metrics, "test": test_intent_metrics},
            fh, indent=2,
        )
    print(f"  [OK] Intent eval  -> {intent_eval_path}")

    priority_eval_path = output_dir / "priority_evaluation.json"
    with priority_eval_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"validation": val_priority_metrics, "test": test_priority_metrics},
            fh, indent=2,
        )
    print(f"  [OK] Priority eval -> {priority_eval_path}")

    print("\n" + "=" * 62)
    print("  STEP 9: Save/Load Verification")
    print("=" * 62)
    # Verify artifacts are loadable and produce identical predictions
    clf2 = MultiOutputClassifier.load(intent_model_path, priority_model_path)
    sample = test_ex[:5]
    preds_before = clf.predict(sample)
    preds_after = clf2.predict(sample)
    match = all(
        p1["intent"] == p2["intent"] and p1["priority"] == p2["priority"]
        for p1, p2 in zip(preds_before, preds_after)
    )
    if match:
        print(f"  [OK] Save/load verification passed (sample n={len(sample)})")
    else:
        print(f"  [FAIL] Save/load mismatch! Before={preds_before[:2]} After={preds_after[:2]}")

    # Final summary
    report = {
        "status": "success",
        "metadata": metadata,
        "validation": {
            "intent": val_intent_metrics,
            "priority": val_priority_metrics,
        },
        "test": {
            "intent": test_intent_metrics,
            "priority": test_priority_metrics,
        },
        "intent_priority_crosstab": crosstab,
        "source_breakdown": source_breakdown,
        "synthetic_breakdown": synthetic_breakdown,
        "save_load_verified": match,
        "timing": {
            "split_seconds": round(split_time, 2),
            "fit_seconds": round(fit_time, 2),
            "val_inference_seconds": round(val_time, 2),
            "test_inference_seconds": round(test_time, 2),
        },
    }
    return report


# ---------------------------------------------------------------------------
# Legacy train_pipeline — for backward compatibility
# ---------------------------------------------------------------------------

def train_pipeline(
    data_path: Path,
    output_model_path: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Legacy JSONL-based training pipeline (kept for backward compatibility).

    For the new XLSX-based pipeline, use main() or run_training_pipeline().
    """
    text_content = data_path.read_text(encoding="utf-8").strip()
    records = []
    if text_content.startswith("["):
        records = json.loads(text_content)
    else:
        lines = [ln.strip() for ln in text_content.splitlines() if ln.strip()]
        records = [json.loads(ln) for ln in lines]

    train_examples = [CanonicalEmailExample.from_dict(r) for r in records]
    clf = MultiOutputClassifier(seed=seed)
    clf.fit(train_examples)
    clf.save_intent(output_model_path.parent / "intent_tfidf.joblib")
    clf.save_priority(output_model_path.parent / "priority_tfidf.joblib")

    return {
        "status": "success",
        "model_identifier": "tfidf-multi-output-logistic-regression",
        "train_examples_count": len(train_examples),
        "intent_classes": list(clf.intent_head.classes_),
        "priority_classes": list(clf.priority_head.classes_),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Inbox AI — Multi-Output TF-IDF Baseline Training Pipeline"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset: .xlsx (smart_inbox_ai_dataset_v2.xlsx) or .jsonl"
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts",
        help="Directory to save all model artifacts and reports"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-intent-model", type=str, default=None,
        help="Override intent model path (default: <output-dir>/intent_tfidf.joblib)"
    )
    parser.add_argument(
        "--output-priority-model", type=str, default=None,
        help="Override priority model path (default: <output-dir>/priority_tfidf.joblib)"
    )
    parser.add_argument(
        "--output-dataset", type=str, default=None,
        help="Override dataset json path (default: <output-dir>/canonical_multi_output_dataset.json)"
    )
    parser.add_argument(
        "--no-print-audit", action="store_true",
        help="Suppress the pre-training audit printout"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    print(f"\nSmart Inbox AI — Multi-Output Training Pipeline")
    print(f"Data   : {data_path}")
    print(f"OutDir : {output_dir}")
    print(f"Seed   : {args.seed}")

    # Load data
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        from ml.v2_dataset_loader import load_v2_dataset
        examples, audit = load_v2_dataset(
            data_path, print_audit=not args.no_print_audit
        )
    elif data_path.suffix.lower() in {".jsonl", ".json"}:
        from ml.v2_dataset_loader import load_from_jsonl
        examples = load_from_jsonl(data_path)
        audit = None
        print(f"Loaded {len(examples)} examples from JSONL")
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}. Use .xlsx or .jsonl")

    if len(examples) < 30:
        raise ValueError(
            f"Only {len(examples)} valid examples found. Refusing to train with fewer than 30."
        )

    # Run pipeline
    report = run_training_pipeline(
        examples=examples,
        output_dir=output_dir,
        data_path=data_path,
        seed=args.seed,
    )

    # Override output paths if specified
    if args.output_intent_model:
        import shutil
        shutil.copy(output_dir / "intent_tfidf.joblib", args.output_intent_model)
    if args.output_priority_model:
        import shutil
        shutil.copy(output_dir / "priority_tfidf.joblib", args.output_priority_model)
    if args.output_dataset:
        import shutil
        shutil.copy(output_dir / "canonical_multi_output_dataset.json", args.output_dataset)

    # Print final summary
    test_intent = report["test"]["intent"]
    test_priority = report["test"]["priority"]
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"  Examples  : {report['metadata']['dataset']['total_examples']}")
    print(f"  Train/Val/Test: "
          f"{report['metadata']['split_counts']['train']}/"
          f"{report['metadata']['split_counts']['val']}/"
          f"{report['metadata']['split_counts']['test']}")
    print(f"  INTENT  TEST: accuracy={test_intent['accuracy']:.4f} "
          f"macro_f1={test_intent['macro_f1']:.4f}")
    print(f"  PRIORITY TEST: accuracy={test_priority['accuracy']:.4f} "
          f"macro_f1={test_priority['macro_f1']:.4f}")
    print(f"  Save/load verified: {report['save_load_verified']}")
    print(f"\n  Artifacts saved to: {output_dir}/")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
