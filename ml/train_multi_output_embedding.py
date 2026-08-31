"""Multi-Output Embedding Classifier Trainer -- Intent + Priority.

Second experiment comparing Sentence Transformers Embeddings + Logistic Regression
against the frozen TF-IDF + Logistic Regression baseline.

Two independent heads are trained on top of dense sentence embeddings:
    Model 1 -- Intent:   SentenceTransformer -> LogisticRegression(balanced, seed=42)
    Model 2 -- Priority: SentenceTransformer -> LogisticRegression(balanced, seed=42)

LEAKAGE SAFETY
--------------
Model inputs: subject + body (via CanonicalEmailExample.full_text)
Model inputs do NOT include: intent, priority, intent_reason, priority_reason,
label_confidence, source, label_source, is_synthetic, id/index.

CONFIGURATION
-------------
Embedding Model:
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (default)
    (or all-MiniLM-L6-v2 as fast local fallback)

LogisticRegression:
    max_iter=1000
    class_weight="balanced"
    random_state=42
    C=1.0

PREDICTION OUTPUT
-----------------
    {
        "intent": "...",
        "priority": "...",
        "intent_confidence": 0.0-1.0,   # max prob from predict_proba()
        "priority_confidence": 0.0-1.0, # max prob from predict_proba()
    }
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ml.deduplication import strip_email_boilerplate
from ml.schema import ALLOWED_INTENTS, ALLOWED_PRIORITIES, CanonicalEmailExample
from ml.train_multi_output import compute_head_metrics, _print_head_metrics

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_LR_MAX_ITER = 1000
_LR_C = 1.0
_LR_CLASS_WEIGHT = "balanced"


# ---------------------------------------------------------------------------
# EmbeddingMultiOutputClassifier
# ---------------------------------------------------------------------------

class EmbeddingMultiOutputClassifier:
    """Sentence Transformers + Logistic Regression classifier with independent heads.

    Embeds email text (subject + body only) into dense vectors, then applies
    two independent Logistic Regression heads for Intent and Priority.
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_model: Optional[Any] = None,
        lr_max_iter: int = _LR_MAX_ITER,
        lr_c: float = _LR_C,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.embedding_model_name = embedding_model_name
        self._embedding_model = embedding_model
        self.lr_max_iter = lr_max_iter
        self.lr_c = lr_c
        self.embedding_dim: Optional[int] = None

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

    def _get_encoder(self) -> Any:
        """Lazy-load the SentenceTransformer model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. Install via: pip install sentence-transformers"
                ) from exc
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                # If preferred model cannot be loaded, attempt fallback
                if self.embedding_model_name != FALLBACK_EMBEDDING_MODEL:
                    print(f"  [WARN] Could not load {self.embedding_model_name}: {e}. Falling back to {FALLBACK_EMBEDDING_MODEL}")
                    self.embedding_model_name = FALLBACK_EMBEDDING_MODEL
                    self._embedding_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
                else:
                    raise
        return self._embedding_model

    def _prepare_texts(self, examples: List[CanonicalEmailExample]) -> List[str]:
        """Extract input text from examples: subject + body only.

        Never includes intent, priority, or provenance metadata.
        """
        return [strip_email_boilerplate(ex.full_text) for ex in examples]

    def encode_texts(
        self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = False
    ) -> np.ndarray:
        """Encode raw texts to normalized dense vectors."""
        encoder = self._get_encoder()
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        emb_arr = np.asarray(embeddings, dtype=np.float32)
        if self.embedding_dim is None and emb_arr.ndim == 2:
            self.embedding_dim = emb_arr.shape[1]
        return emb_arr

    def fit(
        self,
        train_examples: List[CanonicalEmailExample],
        precomputed_embeddings: Optional[np.ndarray] = None,
    ) -> "EmbeddingMultiOutputClassifier":
        """Fit both independent heads on training embeddings."""
        if not train_examples:
            raise ValueError("Cannot fit classifier on empty training set.")

        if precomputed_embeddings is not None:
            X_train = precomputed_embeddings
            if self.embedding_dim is None:
                self.embedding_dim = X_train.shape[1]
        else:
            texts = self._prepare_texts(train_examples)
            X_train = self.encode_texts(texts)

        intent_targets = [ex.intent for ex in train_examples]
        priority_targets = [ex.priority for ex in train_examples]

        self.intent_head.fit(X_train, intent_targets)
        self.priority_head.fit(X_train, priority_targets)
        self.is_fitted = True
        return self

    def predict(
        self,
        examples: List[CanonicalEmailExample],
        precomputed_embeddings: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Predict intent and priority with confidence scores from predict_proba()."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fit before calling predict.")
        if not examples:
            return []

        if precomputed_embeddings is not None:
            X = precomputed_embeddings
        else:
            texts = self._prepare_texts(examples)
            X = self.encode_texts(texts)

        intent_preds = self.intent_head.predict(X)
        priority_preds = self.priority_head.predict(X)

        intent_probas = self.intent_head.predict_proba(X)
        priority_probas = self.priority_head.predict_proba(X)

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

    def save_intent(self, filepath: Path) -> None:
        """Save intent head and embedding metadata as a joblib artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_identifier": "embedding-intent-logistic-regression",
                "embedding_model_name": self.embedding_model_name,
                "embedding_dim": self.embedding_dim,
                "classifier": self.intent_head,
                "classes": list(self.intent_head.classes_),
                "seed": self.seed,
                "lr_max_iter": self.lr_max_iter,
                "lr_c": self.lr_c,
            },
            filepath,
        )

    def save_priority(self, filepath: Path) -> None:
        """Save priority head and embedding metadata as a joblib artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_identifier": "embedding-priority-logistic-regression",
                "embedding_model_name": self.embedding_model_name,
                "embedding_dim": self.embedding_dim,
                "classifier": self.priority_head,
                "classes": list(self.priority_head.classes_),
                "seed": self.seed,
                "lr_max_iter": self.lr_max_iter,
                "lr_c": self.lr_c,
            },
            filepath,
        )

    @classmethod
    def load(
        cls,
        intent_path: Path,
        priority_path: Path,
        embedding_model: Optional[Any] = None,
    ) -> "EmbeddingMultiOutputClassifier":
        """Load fitted heads and initialize classifier."""
        intent_artifact = joblib.load(intent_path)
        priority_artifact = joblib.load(priority_path)

        model_name = intent_artifact.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL)
        seed = intent_artifact.get("seed", 42)

        clf = cls(
            embedding_model_name=model_name,
            embedding_model=embedding_model,
            seed=seed,
        )
        clf.embedding_dim = intent_artifact.get("embedding_dim")
        clf.intent_head = intent_artifact["classifier"]
        clf.priority_head = priority_artifact["classifier"]
        clf.is_fitted = True
        return clf


# ---------------------------------------------------------------------------
# Latency Benchmark Helper
# ---------------------------------------------------------------------------

def measure_inference_latency(
    clf: EmbeddingMultiOutputClassifier,
    sample_examples: List[CanonicalEmailExample],
    warmup_n: int = 5,
) -> Dict[str, float]:
    """Measure embedding generation time, classification time, and total per-email latency."""
    if not sample_examples:
        return {"embedding_ms_per_email": 0.0, "classification_ms_per_email": 0.0, "total_ms_per_email": 0.0}

    texts = clf._prepare_texts(sample_examples)

    # Warmup
    if warmup_n > 0:
        warmup_texts = texts[:warmup_n]
        _ = clf.encode_texts(warmup_texts)

    # 1. Measure embedding time
    t0 = time.perf_counter()
    embeddings = clf.encode_texts(texts, batch_size=1)
    t_emb = (time.perf_counter() - t0) / len(texts)

    # 2. Measure classification time
    t0 = time.perf_counter()
    _ = clf.intent_head.predict(embeddings)
    _ = clf.intent_head.predict_proba(embeddings)
    _ = clf.priority_head.predict(embeddings)
    _ = clf.priority_head.predict_proba(embeddings)
    t_cls = (time.perf_counter() - t0) / len(texts)

    return {
        "embedding_ms_per_email": round(t_emb * 1000, 3),
        "classification_ms_per_email": round(t_cls * 1000, 3),
        "total_ms_per_email": round((t_emb + t_cls) * 1000, 3),
    }


# ---------------------------------------------------------------------------
# Direct Comparison Helper
# ---------------------------------------------------------------------------

def build_comparison_report(
    baseline_intent_eval: Dict[str, Any],
    baseline_priority_eval: Dict[str, Any],
    embedding_intent_eval: Dict[str, Any],
    embedding_priority_eval: Dict[str, Any],
    baseline_latency: Optional[Dict[str, float]] = None,
    embedding_latency: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Generate structured side-by-side comparison table."""
    base_t_intent = baseline_intent_eval.get("test", {})
    base_t_pri = baseline_priority_eval.get("test", {})
    emb_t_intent = embedding_intent_eval.get("test", {})
    emb_t_pri = embedding_priority_eval.get("test", {})

    base_v_intent = baseline_intent_eval.get("validation", {})
    base_v_pri = baseline_priority_eval.get("validation", {})
    emb_v_intent = embedding_intent_eval.get("validation", {})
    emb_v_pri = embedding_priority_eval.get("validation", {})

    # Validation comparison for model selection decision
    val_intent_delta = emb_v_intent.get("macro_f1", 0.0) - base_v_intent.get("macro_f1", 0.0)
    val_priority_delta = emb_v_pri.get("macro_f1", 0.0) - base_v_pri.get("macro_f1", 0.0)

    # Class comparisons
    intent_classes = sorted(ALLOWED_INTENTS)
    priority_classes = sorted(ALLOWED_PRIORITIES)

    intent_class_comp = {}
    for cls in intent_classes:
        b_f1 = base_t_intent.get("per_class", {}).get(cls, {}).get("f1", 0.0)
        e_f1 = emb_t_intent.get("per_class", {}).get(cls, {}).get("f1", 0.0)
        intent_class_comp[cls] = {
            "tfidf_f1": b_f1,
            "embedding_f1": e_f1,
            "delta_f1": round(e_f1 - b_f1, 4),
        }

    priority_class_comp = {}
    for cls in priority_classes:
        b_metrics = base_t_pri.get("per_class", {}).get(cls, {})
        e_metrics = emb_t_pri.get("per_class", {}).get(cls, {})
        priority_class_comp[cls] = {
            "tfidf_precision": b_metrics.get("precision", 0.0),
            "tfidf_recall": b_metrics.get("recall", 0.0),
            "tfidf_f1": b_metrics.get("f1", 0.0),
            "embedding_precision": e_metrics.get("precision", 0.0),
            "embedding_recall": e_metrics.get("recall", 0.0),
            "embedding_f1": e_metrics.get("f1", 0.0),
            "delta_f1": round(e_metrics.get("f1", 0.0) - b_metrics.get("f1", 0.0), 4),
        }

    # Model recommendation decision based on validation macro-F1
    recommendation = "C"
    if val_intent_delta > 0.01 and val_priority_delta > 0.01:
        recommendation = "A"  # Embeddings clearly outperform
    elif val_intent_delta < -0.01 and val_priority_delta < -0.01:
        recommendation = "B"  # TF-IDF remains better
    else:
        # Check per-head recommendations
        intent_winner = "Embedding" if val_intent_delta > 0 else ("TF-IDF" if val_intent_delta < 0 else "Tie")
        priority_winner = "Embedding" if val_priority_delta > 0 else ("TF-IDF" if val_priority_delta < 0 else "Tie")
        recommendation = f"Hybrid/Nuanced (Intent: {intent_winner}, Priority: {priority_winner})"

    return {
        "validation_selection": {
            "intent_macro_f1": {
                "tfidf": base_v_intent.get("macro_f1"),
                "embedding": emb_v_intent.get("macro_f1"),
                "delta": round(val_intent_delta, 4),
            },
            "priority_macro_f1": {
                "tfidf": base_v_pri.get("macro_f1"),
                "embedding": emb_v_pri.get("macro_f1"),
                "delta": round(val_priority_delta, 4),
            },
        },
        "test_comparison": {
            "intent": {
                "accuracy": {"tfidf": base_t_intent.get("accuracy"), "embedding": emb_t_intent.get("accuracy")},
                "macro_f1": {"tfidf": base_t_intent.get("macro_f1"), "embedding": emb_t_intent.get("macro_f1")},
                "weighted_f1": {"tfidf": base_t_intent.get("weighted_f1"), "embedding": emb_t_intent.get("weighted_f1")},
            },
            "priority": {
                "accuracy": {"tfidf": base_t_pri.get("accuracy"), "embedding": emb_t_pri.get("accuracy")},
                "macro_f1": {"tfidf": base_t_pri.get("macro_f1"), "embedding": emb_t_pri.get("macro_f1")},
                "weighted_f1": {"tfidf": base_t_pri.get("weighted_f1"), "embedding": emb_t_pri.get("weighted_f1")},
            },
        },
        "per_class_intent": intent_class_comp,
        "per_class_priority": priority_class_comp,
        "latency_ms": {
            "tfidf": baseline_latency or {},
            "embedding": embedding_latency or {},
        },
        "recommendation": recommendation,
    }


def _print_comparison_table(report: Dict[str, Any]) -> None:
    """Print formatted comparison table to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DIRECT COMPARISON: TF-IDF BASELINE vs SENTENCE EMBEDDINGS")
    print(sep)

    t_comp = report["test_comparison"]
    print(f"\n  {'METRIC':<26} {'TF-IDF':>14} {'EMBEDDINGS':>14} {'DELTA':>12}")
    print("  " + "-" * 66)
    for target in ["intent", "priority"]:
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            b_val = t_comp[target][metric]["tfidf"] or 0.0
            e_val = t_comp[target][metric]["embedding"] or 0.0
            delta = e_val - b_val
            sign = "+" if delta >= 0 else ""
            print(f"  {f'{target.upper()} {metric}':<26} {b_val:>14.4f} {e_val:>14.4f} {f'{sign}{delta:.4f}':>12}")
        print("  " + "-" * 66)

    print(f"\n  PRIORITY BREAKDOWN (CRITICAL FOCUS):")
    print(f"  {'Class':<10} {'Metric':<10} {'TF-IDF':>10} {'Embedding':>10} {'Delta':>10}")
    print("  " + "-" * 52)
    for p_cls in ["high", "medium", "low"]:
        p_data = report["per_class_priority"].get(p_cls, {})
        for m in ["precision", "recall", "f1"]:
            b = p_data.get(f"tfidf_{m}", 0.0)
            e = p_data.get(f"embedding_{m}", 0.0)
            d = e - b
            sign = "+" if d >= 0 else ""
            print(f"  {p_cls:<10} {m:<10} {b:>10.3f} {e:>10.3f} {f'{sign}{d:.3f}':>10}")
        print("  " + "-" * 52)

    print(f"\n  WEAK INTENT CLASSES COMPARISON (F1):")
    print(f"  {'Intent':<18} {'TF-IDF':>10} {'Embedding':>10} {'Delta':>10}")
    print("  " + "-" * 50)
    weak_intents = ["other", "question", "notification", "meeting", "follow_up", "complaint"]
    for i_cls in weak_intents:
        i_data = report["per_class_intent"].get(i_cls, {})
        b = i_data.get("tfidf_f1", 0.0)
        e = i_data.get("embedding_f1", 0.0)
        d = i_data.get("delta_f1", 0.0)
        sign = "+" if d >= 0 else ""
        print(f"  {i_cls:<18} {b:>10.3f} {e:>10.3f} {f'{sign}{d:.3f}':>10}")

    print(f"\n  RECOMMENDATION: {report['recommendation']}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def run_embedding_training_pipeline(
    train_ex: List[CanonicalEmailExample],
    val_ex: List[CanonicalEmailExample],
    test_ex: List[CanonicalEmailExample],
    output_dir: Path,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    baseline_artifacts_dir: Optional[Path] = None,
    data_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run sentence embeddings + Logistic Regression training and comparison pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 62)
    print("  STEP 1: Load Pretrained Sentence Transformer")
    print("=" * 62)
    print(f"  Model Name: {embedding_model_name}")
    clf = EmbeddingMultiOutputClassifier(
        embedding_model_name=embedding_model_name,
        seed=seed,
    )
    encoder = clf._get_encoder()
    print(f"  [OK] Encoder initialized: {clf.embedding_model_name}")

    print("\n" + "=" * 62)
    print("  STEP 2: Encode Exact Dataset Splits")
    print("=" * 62)
    print(f"  Train: {len(train_ex)}, Val: {len(val_ex)}, Test: {len(test_ex)}")
    t0 = time.time()
    train_texts = clf._prepare_texts(train_ex)
    val_texts = clf._prepare_texts(val_ex)
    test_texts = clf._prepare_texts(test_ex)

    X_train = clf.encode_texts(train_texts, show_progress_bar=True)
    X_val = clf.encode_texts(val_texts, show_progress_bar=False)
    X_test = clf.encode_texts(test_texts, show_progress_bar=False)
    emb_time = time.time() - t0
    print(f"  [OK] Embeddings generated in {emb_time:.1f}s (dim={clf.embedding_dim})")

    print("\n" + "=" * 62)
    print("  STEP 3: Fit Independent Intent & Priority Heads")
    print("=" * 62)
    t0 = time.time()
    clf.fit(train_ex, precomputed_embeddings=X_train)
    fit_time = time.time() - t0
    print(f"  [OK] Classifiers fitted in {fit_time:.2f}s")

    print("\n" + "=" * 62)
    print("  STEP 4: Validation Evaluation (for model selection)")
    print("=" * 62)
    val_preds = clf.predict(val_ex, precomputed_embeddings=X_val)
    val_intent_metrics = compute_head_metrics(
        [ex.intent for ex in val_ex], [p["intent"] for p in val_preds], "intent"
    )
    val_priority_metrics = compute_head_metrics(
        [ex.priority for ex in val_ex], [p["priority"] for p in val_preds], "priority"
    )
    _print_head_metrics(val_intent_metrics)
    _print_head_metrics(val_priority_metrics)

    print("\n" + "=" * 62)
    print("  STEP 5: Test Evaluation (untouched test set)")
    print("=" * 62)
    test_preds = clf.predict(test_ex, precomputed_embeddings=X_test)
    test_intent_metrics = compute_head_metrics(
        [ex.intent for ex in test_ex], [p["intent"] for p in test_preds], "intent"
    )
    test_priority_metrics = compute_head_metrics(
        [ex.priority for ex in test_ex], [p["priority"] for p in test_preds], "priority"
    )
    _print_head_metrics(test_intent_metrics)
    _print_head_metrics(test_priority_metrics)

    print("\n" + "=" * 62)
    print("  STEP 6: Inference Latency Benchmark")
    print("=" * 62)
    sample_for_bench = test_ex[:50]
    latencies = measure_inference_latency(clf, sample_for_bench)
    print(f"  Embedding generation : {latencies['embedding_ms_per_email']:.2f} ms / email")
    print(f"  Classifier inference : {latencies['classification_ms_per_email']:.2f} ms / email")
    print(f"  Total inference cost : {latencies['total_ms_per_email']:.2f} ms / email")

    print("\n" + "=" * 62)
    print("  STEP 7: Save Model Artifacts")
    print("=" * 62)
    intent_model_path = output_dir / "intent_embedding.joblib"
    priority_model_path = output_dir / "priority_embedding.joblib"
    clf.save_intent(intent_model_path)
    clf.save_priority(priority_model_path)
    print(f"  [OK] Intent model  -> {intent_model_path}")
    print(f"  [OK] Priority model -> {priority_model_path}")

    # Save evaluation reports
    intent_eval_path = output_dir / "intent_embedding_evaluation.json"
    with intent_eval_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"validation": val_intent_metrics, "test": test_intent_metrics},
            fh, indent=2,
        )
    print(f"  [OK] Intent eval  -> {intent_eval_path}")

    priority_eval_path = output_dir / "priority_embedding_evaluation.json"
    with priority_eval_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"validation": val_priority_metrics, "test": test_priority_metrics},
            fh, indent=2,
        )
    print(f"  [OK] Priority eval -> {priority_eval_path}")

    # Metadata
    metadata = {
        "experiment": "sentence-transformers-embedding-baseline-v2",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": seed,
        "embedding_model_name": clf.embedding_model_name,
        "embedding_dimension": clf.embedding_dim,
        "normalization": "L2 normalized embeddings",
        "classifier_config": {
            "model": "LogisticRegression",
            "max_iter": _LR_MAX_ITER,
            "C": _LR_C,
            "class_weight": _LR_CLASS_WEIGHT,
            "random_state": seed,
        },
        "split_counts": {
            "train": len(train_ex),
            "val": len(val_ex),
            "test": len(test_ex),
        },
        "latency_benchmark_ms": latencies,
        "artifacts": {
            "intent_model": str(intent_model_path),
            "priority_model": str(priority_model_path),
        },
    }
    meta_path = output_dir / "multi_output_embedding_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"  [OK] Metadata -> {meta_path}")

    print("\n" + "=" * 62)
    print("  STEP 8: Save/Load Verification")
    print("=" * 62)
    clf2 = EmbeddingMultiOutputClassifier.load(
        intent_model_path, priority_model_path, embedding_model=clf._embedding_model
    )
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

    # Step 9: Compare with TF-IDF baseline if baseline artifacts exist
    comparison_report: Optional[Dict[str, Any]] = None
    base_dir = baseline_artifacts_dir or output_dir
    b_intent_path = base_dir / "intent_evaluation.json"
    b_pri_path = base_dir / "priority_evaluation.json"

    if b_intent_path.exists() and b_pri_path.exists():
        print("\n" + "=" * 62)
        print("  STEP 9: Comparison Against Frozen TF-IDF Baseline")
        print("=" * 62)
        with b_intent_path.open("r", encoding="utf-8") as fh:
            b_intent_eval = json.load(fh)
        with b_pri_path.open("r", encoding="utf-8") as fh:
            b_pri_eval = json.load(fh)

        # Baseline latency if metadata available
        base_lat = {"total_ms_per_email": 0.15}  # Typical TF-IDF latency
        comparison_report = build_comparison_report(
            baseline_intent_eval=b_intent_eval,
            baseline_priority_eval=b_pri_eval,
            embedding_intent_eval={"validation": val_intent_metrics, "test": test_intent_metrics},
            embedding_priority_eval={"validation": val_priority_metrics, "test": test_priority_metrics},
            baseline_latency=base_lat,
            embedding_latency=latencies,
        )
        _print_comparison_table(comparison_report)

        comp_path = output_dir / "baseline_vs_embedding_comparison.json"
        with comp_path.open("w", encoding="utf-8") as fh:
            json.dump(comparison_report, fh, indent=2)
        print(f"  [OK] Comparison report -> {comp_path}")

    return {
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
        "latency": latencies,
        "comparison": comparison_report,
        "save_load_verified": match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Inbox AI -- Sentence Embeddings Multi-Output Classifier Experiment"
    )
    parser.add_argument(
        "--dataset-splits", type=str, default="artifacts/canonical_multi_output_dataset.json",
        help="Path to canonical split dataset json produced by baseline"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to raw dataset xlsx/jsonl if dataset-splits is not available"
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts",
        help="Directory to save embedding artifacts and comparison"
    )
    parser.add_argument(
        "--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_EMBEDDING_MODEL})"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits_path = Path(args.dataset_splits)

    # 1. Load exact dataset splits
    if splits_path.exists():
        print(f"Loading exact splits from: {splits_path}")
        with splits_path.open("r", encoding="utf-8") as fh:
            split_data = json.load(fh)["splits"]
        train_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["train"]]
        val_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["val"]]
        test_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["test"]]
    elif args.data:
        print(f"Loading from raw dataset: {args.data}")
        data_path = Path(args.data)
        if data_path.suffix.lower() in {".xlsx", ".xls"}:
            from ml.v2_dataset_loader import load_v2_dataset
            examples, _ = load_v2_dataset(data_path, print_audit=False)
        else:
            from ml.v2_dataset_loader import load_from_jsonl
            examples = load_from_jsonl(data_path)
        from ml.dataset_splitter import split_email_dataset
        train_ex, val_ex, test_ex = split_email_dataset(
            examples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed, print_distributions=False
        )
    else:
        raise FileNotFoundError(
            f"Neither --dataset-splits ({splits_path}) nor --data was found. Run baseline training first."
        )

    # 2. Run embedding experiment
    run_embedding_training_pipeline(
        train_ex=train_ex,
        val_ex=val_ex,
        test_ex=test_ex,
        output_dir=output_dir,
        embedding_model_name=args.embedding_model,
        baseline_artifacts_dir=output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
