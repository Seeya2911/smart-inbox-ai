"""Unit tests for the Multi-Output Priority + Intent Training and Labeling Pipeline."""
from __future__ import annotations

from pathlib import Path
import pytest

from ml.deduplication import deduplicate_dataset, strip_email_boilerplate
from ml.generate_synthetic import generate_synthetic_examples
from ml.gold_labeler import create_gold_splits, load_gold_split
from ml.intent_rules import IntentRuleEngine
from ml.schema import CanonicalEmailExample, format_namespaced_id
from ml.train_multi_output import MultiOutputClassifier
from ml.weak_labeler import DualWeakLabeler


def test_schema_validation_and_id_namespacing():
    """Verify CanonicalEmailExample validation and ID namespacing."""
    ex = CanonicalEmailExample(
        id="enron_00123",
        subject="Quarterly Budget Review",
        body="Please review the attached financial audit details.",
        intent="request",
        priority="high",
        source="enron",
        label_source="human",
        label_confidence=1.0,
    )
    assert ex.id == "enron_00123"
    assert ex.intent == "request"
    assert ex.priority == "high"

    # Test namespacing helper
    assert format_namespaced_id("enron", "101") == "enron_101"
    assert format_namespaced_id("spam", "spam_202") == "spam_202"

    # Invalid ID namespace raises ValueError
    with pytest.raises(ValueError, match="Example id 'invalid_id_format' must be namespaced"):
        CanonicalEmailExample(
            id="invalid_id_format",
            subject="Test",
            body="Body text",
            intent="request",
            priority="low",
        )


def test_boilerplate_stripping_and_deduplication():
    """Verify email signature, quoted reply, and forward header stripping, plus deduplication."""
    raw_email = (
        "---------- Forwarded message ----------\n"
        "From: Alice <alice@corp.com>\n\n"
        "Please review the document by 5 PM.\n\n"
        "> On Monday, Bob wrote:\n"
        "> Previous discussion about meeting details.\n\n"
        "Thanks,\nAlice Smith\nTech Corp"
    )
    clean = strip_email_boilerplate(raw_email)
    assert "forwarded message" not in clean
    assert "previous discussion" not in clean
    assert "alice smith" not in clean
    assert "please review the document by 5 pm" in clean

    ex1 = CanonicalEmailExample.from_dict({"id": "enron_001", "body": raw_email, "intent": "request", "priority": "high", "source": "enron"})
    ex2 = CanonicalEmailExample.from_dict({"id": "enron_002", "body": "Please review the document by 5 PM.", "intent": "request", "priority": "high", "source": "enron"})

    deduped, stats = deduplicate_dataset([ex1, ex2])
    assert len(deduped) == 1
    assert stats["total_removed"] == 1


def test_dual_weak_labeler_and_score_routing():
    """Verify weak labeling and 3-population routing."""
    labeler = DualWeakLabeler(high_threshold=3.0, low_threshold=1.0)
    high_req = {
        "id": "synthetic_001",
        "subject": "URGENT: Password reset required immediately",
        "body": "Security alert: suspicious login attempt detected. Verify your account code 482910.",
        "source": "synthetic",
    }
    ex = labeler.evaluate_email(high_req)
    assert ex.intent == "security"
    assert ex.priority == "high"
    assert ex.rule_score >= 3.0

    high, amb, low = labeler.route_populations([ex])
    assert len(high) == 1
    assert len(amb) == 0
    assert len(low) == 0


def test_noisy_synthetic_gap_generator():
    """Verify targeted noisy synthetic gap generator."""
    syn_examples = generate_synthetic_examples(count=10, seed=42)
    assert len(syn_examples) == 10
    for ex in syn_examples:
        assert ex.id.startswith("synthetic_")
        assert ex.source == "synthetic"
        assert ex.is_synthetic is True


def test_multi_output_tfidf_baseline_training():
    """Verify fitting TF-IDF baseline for Intent Head and Priority Head independently."""
    train_data = [
        CanonicalEmailExample.from_dict({"id": "enron_01", "subject": "2FA Alert", "body": "Your security code is 987123", "intent": "security", "priority": "high", "source": "enron"}),
        CanonicalEmailExample.from_dict({"id": "enron_02", "subject": "Order Shipped", "body": "Your invoice receipt order #4812 has shipped", "intent": "transactional", "priority": "low", "source": "enron"}),
        CanonicalEmailExample.from_dict({"id": "enron_03", "subject": "Meeting Friday", "body": "Let us schedule a zoom call for Friday", "intent": "meeting", "priority": "medium", "source": "enron"}),
    ]

    clf = MultiOutputClassifier(seed=42)
    clf.fit(train_data)
    assert clf.is_fitted is True

    preds = clf.predict(train_data)
    assert len(preds) == 3
    for p in preds:
        assert "intent" in p
        assert "priority" in p


def test_gold_test_set_isolation_invariant(tmp_path: Path):
    """Invariant Test: Asserts ZERO ID or text overlap between gold/test split and training split."""
    raw_gold = [
        CanonicalEmailExample.from_dict({"id": f"inbox_{i:04d}", "subject": f"Gold Email {i}", "body": f"Hand labeled body content number {i}", "intent": "request" if i % 2 == 0 else "information", "priority": "high" if i % 2 == 0 else "low", "source": "inbox"}).to_dict()
        for i in range(20)
    ]

    gold_examples = [CanonicalEmailExample.from_dict(r) for r in raw_gold]
    summary = create_gold_splits(gold_examples, output_dir=tmp_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    gold_train = load_gold_split(tmp_path / "train.jsonl")
    gold_test = load_gold_split(tmp_path / "test.jsonl")

    train_ids = {ex.id for ex in gold_train}
    test_ids = {ex.id for ex in gold_test}

    train_clean_texts = {strip_email_boilerplate(ex.full_text) for ex in gold_train}
    test_clean_texts = {strip_email_boilerplate(ex.full_text) for ex in gold_test}

    # Strict isolation invariant assertions
    assert not (train_ids & test_ids), "Gold Test Set Leakage Violation: ID overlap detected between train and test splits!"
    assert not (train_clean_texts & test_clean_texts), "Gold Test Set Leakage Violation: Text overlap detected between train and test splits!"
