"""Priority-aware multi-output classification system for Smart Inbox AI.

Combines TF-IDF n-gram features with engineered text signals (urgency,
deadlines, security flags, action imperatives) into a hybrid sparse
representation for the priority head. Supports decision thresholding for
high-priority detection.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.deduplication import strip_email_boilerplate
from ml.priority_features import PriorityFeatureExtractor
from ml.schema import CanonicalEmailExample
from ml.train_multi_output import compute_head_metrics


class PriorityAwareClassifier:
    """Priority-aware classification system with independent Intent and Priority heads.

    Intent Head: Dedicated TF-IDF + LogisticRegression.
    Priority Head: Hybrid representation (TF-IDF + text engineered features) +
                   LogisticRegression with configurable high-priority thresholding.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        sublinear_tf: bool = True,
        c_intent: float = 1.0,
        c_priority: float = 1.0,
        class_weight_priority: Union[str, Dict[str, float]] = "balanced",
        class_weight_intent: str = "balanced",
        high_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.c_intent = c_intent
        self.c_priority = c_priority
        self.class_weight_priority = class_weight_priority
        self.class_weight_intent = class_weight_intent
        self.high_threshold = high_threshold

        self.intent_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.priority_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.priority_feature_extractor = PriorityFeatureExtractor()

        self.intent_head = LogisticRegression(
            C=c_intent,
            max_iter=1000,
            class_weight=class_weight_intent,
            random_state=seed,
        )
        self.priority_head = LogisticRegression(
            C=c_priority,
            max_iter=1000,
            class_weight=class_weight_priority,
            random_state=seed,
        )
        self.is_fitted = False

    def _prepare_texts(self, examples: List[CanonicalEmailExample]) -> List[str]:
        return [strip_email_boilerplate(ex.full_text) for ex in examples]

    def _build_priority_features(self, examples: List[CanonicalEmailExample], fit: bool = False):
        texts = self._prepare_texts(examples)
        if fit:
            X_tfidf = self.priority_vectorizer.fit_transform(texts)
        else:
            X_tfidf = self.priority_vectorizer.transform(texts)
        X_engineered = self.priority_feature_extractor.extract_sparse(examples)
        return hstack([X_tfidf, X_engineered], format="csr")

    def fit(self, train_examples: List[CanonicalEmailExample]) -> "PriorityAwareClassifier":
        texts = self._prepare_texts(train_examples)
        y_intent = [ex.intent for ex in train_examples]
        y_priority = [ex.priority for ex in train_examples]

        X_intent = self.intent_vectorizer.fit_transform(texts)
        X_priority = self._build_priority_features(train_examples, fit=True)

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
        X_priority = self._build_priority_features(examples, fit=False)

        preds_i = self.intent_head.predict(X_intent)
        proba_i = self.intent_head.predict_proba(X_intent)

        proba_p = self.priority_head.predict_proba(X_priority)
        classes_p = list(self.priority_head.classes_)
        high_idx = classes_p.index("high") if "high" in classes_p else -1

        results = []
        for i in range(len(examples)):
            pi = str(preds_i[i])
            conf_i = float(np.max(proba_i[i]))

            p_probs = proba_p[i]
            # Apply high priority threshold policy
            if high_idx >= 0 and p_probs[high_idx] >= self.high_threshold:
                pp = "high"
                conf_p = float(p_probs[high_idx])
            else:
                # Default argmax amongst non-high or all
                best_idx = int(np.argmax(p_probs))
                pp = str(classes_p[best_idx])
                conf_p = float(p_probs[best_idx])

            results.append({
                "intent": pi,
                "priority": pp,
                "intent_confidence": conf_i,
                "priority_confidence": conf_p,
            })

        return results

    def save(self, intent_path: Path, priority_path: Path) -> None:
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_identifier": "smart-inbox-intent-logistic-regression",
                "vectorizer": self.intent_vectorizer,
                "classifier": self.intent_head,
            },
            intent_path,
        )
        joblib.dump(
            {
                "model_identifier": "smart-inbox-priority-aware-classifier",
                "vectorizer": self.priority_vectorizer,
                "feature_extractor": self.priority_feature_extractor,
                "classifier": self.priority_head,
                "high_threshold": self.high_threshold,
                "config": {
                    "ngram_range": self.ngram_range,
                    "max_features": self.max_features,
                    "sublinear_tf": self.sublinear_tf,
                    "c_priority": self.c_priority,
                    "class_weight": self.class_weight_priority,
                    "high_threshold": self.high_threshold,
                },
            },
            priority_path,
        )
