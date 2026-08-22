"""Unit tests for the INTENT supervised NLP training and evaluation pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.data_quality import (
    DataLeakageError,
    DataQualityError,
    check_dataset_integrity,
    check_split_leakage,
)
from ml.dataset_splitter import split_intent_dataset
from ml.intent_classifier import (
    KeywordIntentClassifier,
    TfidfIntentClassifier,
)
from ml.intent_mapping import map_and_filter_dataset, map_raw_record
from ml.schema import CanonicalIntentExample


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "intent_sample.jsonl"


def test_schema_validation_valid():
    """Verify CanonicalIntentExample instantiation with valid fields."""
    ex = CanonicalIntentExample(
        text="Please review the document by 5 PM.",
        language="en",
        canonical_intent="request",
        source_dataset="test_ds",
        source_example_id="101",
        original_label="ACTION_REQUIRED",
    )
    assert ex.text == "Please review the document by 5 PM."
    assert ex.canonical_intent == "request"
    assert ex.language == "en"


def test_schema_validation_rejects_urgency_and_priority():
    """Verify that canonical schema strictly rejects urgency and priority fields."""
    data = {
        "text": "Please review this document",
        "language": "en",
        "canonical_intent": "request",
        "source_dataset": "test_ds",
        "source_example_id": "102",
        "original_label": "ACTION_REQUIRED",
        "urgency": "high",  # Forbidden
    }
    with pytest.raises(ValueError, match="Forbidden fields present in canonical intent schema"):
        CanonicalIntentExample.from_dict(data)


def test_empty_invalid_records():
    """Verify rejection of empty text, unsupported language, or invalid intent."""
    with pytest.raises(ValueError, match="Canonical text must be a non-empty string"):
        CanonicalIntentExample(
            text="   ",
            language="en",
            canonical_intent="request",
            source_dataset="test_ds",
            source_example_id="103",
            original_label="ACTION_REQUIRED",
        )

    with pytest.raises(ValueError, match="Unsupported language"):
        CanonicalIntentExample(
            text="Valid text",
            language="jp",
            canonical_intent="request",
            source_dataset="test_ds",
            source_example_id="104",
            original_label="ACTION_REQUIRED",
        )

    with pytest.raises(ValueError, match="Unsupported canonical_intent"):
        CanonicalIntentExample(
            text="Valid text",
            language="en",
            canonical_intent="invalid_intent_category",
            source_dataset="test_ds",
            source_example_id="105",
            original_label="ACTION_REQUIRED",
        )


def test_label_mapping_defensible_and_exclusions():
    """Verify defensible mapping of source labels and exclusion of ambiguous labels."""
    # Enron ACTION_REQUIRED maps defensibly to request
    enron_pos = {
        "source_example_id": "1",
        "language": "en",
        "text": "Could you send the summary report?",
        "action_intent": "ACTION_REQUIRED",
        "source_dataset": "Charlie9/enron_intent_dataset_verified",
    }
    example_pos, exclusion_pos = map_raw_record(enron_pos)
    assert example_pos is not None
    assert exclusion_pos is None
    assert example_pos.canonical_intent == "request"

    # Enron NO_ACTION_REQUIRED is excluded because mapping is ambiguous
    enron_neg = {
        "source_example_id": "2",
        "language": "en",
        "text": "Thanks for the update.",
        "action_intent": "NO_ACTION_REQUIRED",
        "source_dataset": "Charlie9/enron_intent_dataset_verified",
    }
    example_neg, exclusion_neg = map_raw_record(enron_neg)
    assert example_neg is None
    assert exclusion_neg is not None
    assert exclusion_neg.source_example_id == "2"
    assert "Uncertain mapping" in exclusion_neg.reason


def test_duplicate_detection():
    """Verify quality check raises DataQualityError on duplicate text or source IDs."""
    ex1 = CanonicalIntentExample(
        text="Unique text message 1",
        language="en",
        canonical_intent="request",
        source_dataset="ds1",
        source_example_id="id1",
        original_label="req",
    )
    ex2 = CanonicalIntentExample(
        text="Unique text message 1",  # Duplicate text
        language="en",
        canonical_intent="request",
        source_dataset="ds1",
        source_example_id="id2",
        original_label="req",
    )

    with pytest.raises(DataQualityError, match="Duplicate text detected"):
        check_dataset_integrity([ex1, ex2], allow_internal_duplicates=False)


def test_leakage_detection_fails_loudly():
    """Verify that split leakage detection fails loudly with DataLeakageError."""
    train_ex = [
        CanonicalIntentExample(
            text="Please send me the weekly financial report.",
            language="en",
            canonical_intent="request",
            source_dataset="ds1",
            source_example_id="tr1",
            original_label="req",
        )
    ]
    val_ex = []
    # Test split has near-identical normalized text
    test_ex = [
        CanonicalIntentExample(
            text="please send me the weekly financial report",
            language="en",
            canonical_intent="request",
            source_dataset="ds1",
            source_example_id="te1",
            original_label="req",
        )
    ]

    with pytest.raises(DataLeakageError, match="DATA LEAKAGE DETECTED"):
        check_split_leakage(train_ex, val_ex, test_ex)


def test_deterministic_splitting():
    """Verify that splitting is deterministic given a fixed random seed."""
    distinct_topics = [
        "meeting scheduling details for Monday",
        "financial quarterly budget allocation review",
        "customer inquiry about product delivery status",
        "password reset confirmation notification",
        "promotional discount voucher for subscribers",
        "system maintenance window scheduled at midnight",
        "complaint regarding delayed package shipment",
        "follow up on previous project deadline discussion",
        "request for additional project resource access",
        "invitation to quarterly team retrospective",
        "weekly team digest and announcement summary",
        "urgent request for client account verification",
        "reminder to complete annual compliance training",
        "billing invoice issue resolution steps",
        "technical support ticket status update",
        "survey feedback request for recently completed call",
        "security alert regarding new login attempt",
        "welcome email for new organization members",
        "policy document update announcement",
        "calendar event cancellation notification",
    ]

    records = []
    for i, topic in enumerate(distinct_topics):
        records.append(
            CanonicalIntentExample(
                text=topic,
                language="en",
                canonical_intent="request" if i % 2 == 0 else "information",
                source_dataset="ds",
                source_example_id=str(i),
                original_label="lbl",
            )
        )

    tr1, val1, te1 = split_intent_dataset(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    tr2, val2, te2 = split_intent_dataset(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert [x.source_example_id for x in tr1] == [x.source_example_id for x in tr2]
    assert [x.source_example_id for x in val1] == [x.source_example_id for x in val2]
    assert [x.source_example_id for x in te1] == [x.source_example_id for x in te2]


def test_inference_output_format():
    """Verify classifier prediction and probability formatting."""
    train_data = [
        CanonicalIntentExample("Can you send the doc?", "en", "request", "ds", "1", "req"),
        CanonicalIntentExample("Here is the requested report.", "en", "information", "ds", "2", "info"),
    ]
    test_data = [
        CanonicalIntentExample("Please send the update.", "en", "request", "ds", "3", "req"),
    ]

    clf = TfidfIntentClassifier(seed=42)
    clf.fit(train_data)

    preds = clf.predict(test_data)
    assert len(preds) == 1
    assert isinstance(preds[0], str)

    probs = clf.predict_proba(test_data)
    assert len(probs) == 1
    assert isinstance(probs[0], dict)
    assert sum(probs[0].values()) == pytest.approx(1.0)


def test_fixture_pipeline_offline(monkeypatch):
    """Run full mapping, quality check, splitting, and TF-IDF training on fixture without external calls."""
    assert FIXTURE_PATH.is_file()

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        raw_records = [json.loads(line) for line in handle]

    valid_ex, exclusions, summary = map_and_filter_dataset(raw_records)
    assert len(valid_ex) > 0
    assert summary["excluded_examples_count"] > 0  # Enron NO_ACTION_REQUIRED excluded

    tr, val, te = split_intent_dataset(valid_ex, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    clf = TfidfIntentClassifier(seed=42)
    clf.fit(tr)

    preds = clf.predict(te)
    assert len(preds) == len(te)

    # Legacy Keyword Baseline
    kw_clf = KeywordIntentClassifier()
    kw_preds = kw_clf.predict(te)
    assert len(kw_preds) == len(te)
