"""Multi-Output Baseline Classifier Trainer (TF-IDF + Logistic Regression).

Trains two independent prediction heads (Intent Head and Priority Head) on boilerplate-stripped text.
Commits strictly to TF-IDF + Logistic Regression as the mandatory first baseline.
Applies class weighting (class_weight='balanced') to prevent underfitting on rare high-stakes classes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.deduplication import strip_email_boilerplate
from ml.schema import CanonicalEmailExample


class MultiOutputClassifier:
    """TF-IDF + Logistic Regression Classifier predicting Intent and Priority independently."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.intent_head = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
        self.priority_head = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
        self.is_fitted = False

    def _prepare_texts(self, examples: List[CanonicalEmailExample]) -> List[str]:
        """Extract and clean text bodies from canonical examples."""
        return [strip_email_boilerplate(ex.full_text) for ex in examples]

    def fit(self, train_examples: List[CanonicalEmailExample]) -> MultiOutputClassifier:
        """Fit vectorizer, Intent head, and Priority head strictly on training examples."""
        if not train_examples:
            raise ValueError("Cannot fit classifier on empty training set.")

        texts = self._prepare_texts(train_examples)
        intent_targets = [ex.intent for ex in train_examples]
        priority_targets = [ex.priority for ex in train_examples]

        X_train = self.vectorizer.fit_transform(texts)
        self.intent_head.fit(X_train, intent_targets)
        self.priority_head.fit(X_train, priority_targets)

        self.is_fitted = True
        return self

    def predict(self, examples: List[CanonicalEmailExample]) -> List[Dict[str, str]]:
        """Predict intent and priority for a list of canonical examples.

        Returns list of dicts: [{"intent": "...", "priority": "..."}, ...]
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fit before calling predict.")
        if not examples:
            return []

        texts = self._prepare_texts(examples)
        X = self.vectorizer.transform(texts)

        intent_preds = self.intent_head.predict(X)
        priority_preds = self.priority_head.predict(X)

        return [
            {"intent": str(ip), "priority": str(pp)}
            for ip, pp in zip(intent_preds, priority_preds)
        ]

    def save(self, filepath: Path) -> None:
        """Save model joblib artifact to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model_identifier": "tfidf-multi-output-logistic-regression",
            "vectorizer": self.vectorizer,
            "intent_head": self.intent_head,
            "priority_head": self.priority_head,
            "intent_classes": list(getattr(self.intent_head, "classes_", [])),
            "priority_classes": list(getattr(self.priority_head, "classes_", [])),
            "seed": self.seed,
        }
        joblib.dump(artifact, filepath)

    @classmethod
    def load(cls, filepath: Path) -> MultiOutputClassifier:
        """Load fitted model artifact from disk."""
        artifact = joblib.load(filepath)
        clf = cls(seed=artifact.get("seed", 42))
        clf.vectorizer = artifact["vectorizer"]
        clf.intent_head = artifact["intent_head"]
        clf.priority_head = artifact["priority_head"]
        clf.is_fitted = True
        return clf


def train_pipeline(
    data_path: Path,
    output_model_path: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train multi-output intent + priority baseline model and save artifact."""
    text_content = data_path.read_text(encoding="utf-8").strip()
    records = []
    if text_content.startswith("["):
        records = json.loads(text_content)
    else:
        lines = [l.strip() for l in text_content.splitlines() if l.strip()]
        records = [json.loads(l) for l in lines]

    train_examples = [CanonicalEmailExample.from_dict(r) for r in records]

    clf = MultiOutputClassifier(seed=seed)
    clf.fit(train_examples)
    clf.save(output_model_path)

    summary = {
        "status": "success",
        "model_identifier": "tfidf-multi-output-logistic-regression",
        "model_artifact": str(output_model_path),
        "train_examples_count": len(train_examples),
        "intent_classes": list(clf.intent_head.classes_),
        "priority_classes": list(clf.priority_head.classes_),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Output Baseline Classifier Trainer CLI")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (json/jsonl)")
    parser.add_argument("--output-model", type=str, default="artifacts/multi_output_model.joblib", help="Output model joblib path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    summary = train_pipeline(
        data_path=Path(args.data),
        output_model_path=Path(args.output_model),
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
