"""Intent Classifier Architectures and Baseline Wrappers.

Supports three intent classification models:
1. Legacy Keyword Baseline
2. TF-IDF + Logistic Regression
3. Pretrained Multilingual Encoder (sentence-transformers/paraphrase-multilingual-mpnet-base-v2) + Logistic Regression

Per project requirements, the sentence transformer foundation model is used ONLY as a
pretrained representation layer. The downstream Logistic Regression classifier is trained
by this project. The foundation model itself was NOT created or trained by us.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation.runner import KeywordBaseline
from ml.schema import CanonicalIntentExample

PRETRAINED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class BaseIntentClassifier(ABC):
    """Abstract interface for all intent classification models."""

    @abstractmethod
    def fit(self, examples: List[CanonicalIntentExample]) -> BaseIntentClassifier:
        """Fit model parameters on training examples."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, examples: List[CanonicalIntentExample]) -> List[str]:
        """Predict canonical intent for given examples."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, examples: List[CanonicalIntentExample]) -> List[Dict[str, float]]:
        """Predict intent probability distribution for given examples."""
        raise NotImplementedError


class KeywordIntentClassifier(BaseIntentClassifier):
    """Adapter for the legacy rule/keyword baseline."""

    def __init__(self) -> None:
        self.baseline = KeywordBaseline()

    def fit(self, examples: List[CanonicalIntentExample]) -> KeywordIntentClassifier:
        # Keyword baseline has no trainable parameters
        return self

    def predict(self, examples: List[CanonicalIntentExample]) -> List[str]:
        predictions = []
        for ex in examples:
            pred_dict = self.baseline.predict({"subject": "", "body": ex.text})
            predictions.append(pred_dict.get("intent", "other"))
        return predictions

    def predict_proba(self, examples: List[CanonicalIntentExample]) -> List[Dict[str, float]]:
        preds = self.predict(examples)
        results = []
        for p in preds:
            results.append({p: 1.0})
        return results


class TfidfIntentClassifier(BaseIntentClassifier):
    """TF-IDF Vectorizer + Logistic Regression intent classifier baseline."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", random_state=seed)
        self.classes_: np.ndarray = np.array([])

    def fit(self, examples: List[CanonicalIntentExample]) -> TfidfIntentClassifier:
        texts = [ex.text for ex in examples]
        labels = [ex.canonical_intent for ex in examples]
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.classes_ = np.array(self.clf.classes_)
        return self

    def predict(self, examples: List[CanonicalIntentExample]) -> List[str]:
        texts = [ex.text for ex in examples]
        X = self.vectorizer.transform(texts)
        return list(self.clf.predict(X))

    def predict_proba(self, examples: List[CanonicalIntentExample]) -> List[Dict[str, float]]:
        texts = [ex.text for ex in examples]
        X = self.vectorizer.transform(texts)
        probs = self.clf.predict_proba(X)
        results = []
        for p_row in probs:
            results.append({cls_name: float(p_val) for cls_name, p_val in zip(self.classes_, p_row)})
        return results


class EmbeddingIntentClassifier(BaseIntentClassifier):
    """Pretrained Multilingual Encoder + Logistic Regression intent classifier."""

    def __init__(
        self,
        model_name: str = PRETRAINED_MODEL_ID,
        C: float = 1.0,
        max_iter: int = 1000,
        seed: int = 42,
        encoder_instance: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._encoder = encoder_instance
        self.clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", random_state=seed)
        self.classes_: np.ndarray = np.array([])

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            # Lazy load SentenceTransformer
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        encoder = self._get_encoder()
        embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings)

    def fit(self, examples: List[CanonicalIntentExample]) -> EmbeddingIntentClassifier:
        texts = [ex.text for ex in examples]
        labels = [ex.canonical_intent for ex in examples]
        X = self._encode_texts(texts)
        self.clf.fit(X, labels)
        self.classes_ = np.array(self.clf.classes_)
        return self

    def predict(self, examples: List[CanonicalIntentExample]) -> List[str]:
        texts = [ex.text for ex in examples]
        X = self._encode_texts(texts)
        return list(self.clf.predict(X))

    def predict_proba(self, examples: List[CanonicalIntentExample]) -> List[Dict[str, float]]:
        texts = [ex.text for ex in examples]
        X = self._encode_texts(texts)
        probs = self.clf.predict_proba(X)
        results = []
        for p_row in probs:
            results.append({cls_name: float(p_val) for cls_name, p_val in zip(self.classes_, p_row)})
        return results


def save_intent_model(
    model: BaseIntentClassifier,
    filepath: str,
    metadata: Dict[str, Any],
) -> None:
    """Save trained model and reproducibility metadata using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    payload = {
        "model_class": model.__class__.__name__,
        "model": model,
        "metadata": metadata,
    }
    joblib.dump(payload, filepath)


def load_intent_model(filepath: str) -> Tuple[BaseIntentClassifier, Dict[str, Any]]:
    """Load trained model and reproducibility metadata from joblib file."""
    payload = joblib.load(filepath)
    return payload["model"], payload["metadata"]
