"""Unit tests for priority text feature extraction."""
import numpy as np
import pytest

from ml.priority_features import PriorityFeatureExtractor
from ml.schema import CanonicalEmailExample


def _make_example(ex_id: str, subject: str, body: str) -> CanonicalEmailExample:
    return CanonicalEmailExample(
        id=f"synthetic_{ex_id}",
        subject=subject,
        body=body,
        intent="information",
        priority="low",
    )


class TestPriorityFeatureExtractor:
    def test_extract_features_shape(self) -> None:
        examples = [
            _make_example("1", "Urgent: security alert immediately", "Action required please respond by EOD today."),
            _make_example("2", "Weekly sync meeting agenda", "Here are the notes from last week."),
        ]
        extractor = PriorityFeatureExtractor()
        dense = extractor.extract_features(examples)
        assert dense.shape == (2, len(extractor.FEATURE_NAMES))
        assert dense.dtype == np.float32

    def test_urgent_subject_flag(self) -> None:
        urgent_ex = _make_example("u1", "URGENT: Server down", "Please check")
        normal_ex = _make_example("n1", "Monthly report summary", "Please review")

        extractor = PriorityFeatureExtractor()
        features = extractor.extract_features([urgent_ex, normal_ex])

        # Feature index 0 is subject_has_urgent
        assert features[0, 0] == 1.0
        assert features[1, 0] == 0.0

    def test_deadline_expression_detection(self) -> None:
        deadline_ex = _make_example("d1", "Contract review", "Please submit by EOD tomorrow without delay.")
        features = PriorityFeatureExtractor().extract_features([deadline_ex])

        # Feature index 2 is has_deadline_expr
        assert features[0, 2] == 1.0

    def test_security_and_financial_alerts(self) -> None:
        sec_ex = _make_example("s1", "Security alert: unauthorized access", "Password reset required immediately.")
        fin_ex = _make_example("f1", "Final notice: payment overdue", "Your account is past due.")

        extractor = PriorityFeatureExtractor()
        features = extractor.extract_features([sec_ex, fin_ex])

        # Index 3: has_security_alert, Index 4: has_financial_urgency
        assert features[0, 3] == 1.0
        assert features[1, 4] == 1.0

    def test_extract_sparse_format(self) -> None:
        ex = _make_example("sp1", "Notice", "Body content")
        sparse = PriorityFeatureExtractor().extract_sparse([ex])
        assert sparse.shape == (1, len(PriorityFeatureExtractor.FEATURE_NAMES))
        assert hasattr(sparse, "tocsr")
