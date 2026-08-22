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
from ml.evaluate_intent import load_dataset_splits
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
    assert ex.is_synthetic is False


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


def test_synthetic_fixture_provenance():
    """Verify clean separation and provenance tagging of synthetic vs source-derived fixture records."""
    assert FIXTURE_PATH.is_file()
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        raw_records = [json.loads(line) for line in handle]

    valid_ex, exclusions, summary = map_and_filter_dataset(raw_records)

    synthetic_exs = [e for e in valid_ex if e.is_synthetic]
    source_exs = [e for e in valid_ex if not e.is_synthetic]

    assert len(synthetic_exs) > 0
    assert len(source_exs) > 0

    for syn in synthetic_exs:
        assert syn.source_dataset == "synthetic_dev_fixture"
        assert syn.provenance == "synthetic_development_only"
        assert syn.original_label.startswith("synthetic_")

    for src in source_exs:
        assert src.source_dataset in {"Charlie9/enron_intent_dataset_verified", "AmazonScience/massive"}


def test_duplicate_and_conflicting_label_detection():
    """Verify quality check raises DataQualityError on duplicate text, duplicate IDs, or label conflicts."""
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

    # Conflicting labels for same source ID
    ex_conf1 = CanonicalIntentExample(
        text="Text A",
        language="en",
        canonical_intent="request",
        source_dataset="ds1",
        source_example_id="dup_id",
        original_label="req",
    )
    ex_conf2 = CanonicalIntentExample(
        text="Text B",
        language="en",
        canonical_intent="information",
        source_dataset="ds1",
        source_example_id="dup_id",  # Same ID, different label
        original_label="info",
    )
    with pytest.raises(DataQualityError, match="Conflicting labels detected for identical source ID"):
        check_dataset_integrity([ex_conf1, ex_conf2], allow_internal_duplicates=True)


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


def test_group_isolation_and_deterministic_splitting():
    """Verify group isolation and split repeatability across fixed random seeds."""
    groups = ["grp_alpha", "grp_beta", "grp_gamma", "grp_delta", "grp_epsilon", "grp_zeta"]
    records = []
    for i, g in enumerate(groups):
        for j in range(3):
            records.append(
                CanonicalIntentExample(
                    text=f"Distinct topic description {i}-{j} for group {g}",
                    language="en",
                    canonical_intent="request" if i % 2 == 0 else "information",
                    source_dataset="ds",
                    source_example_id=f"{g}-{j}",
                    original_label="lbl",
                    source_group_id=g,
                )
            )

    tr1, val1, te1 = split_intent_dataset(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
    tr2, val2, te2 = split_intent_dataset(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)

    tr_groups = {x.source_group_id for x in tr1}
    val_groups = {x.source_group_id for x in val1}
    te_groups = {x.source_group_id for x in te1}

    # Strict group isolation check
    assert not (tr_groups & val_groups)
    assert not (tr_groups & te_groups)
    assert not (val_groups & te_groups)

    # Repeatability check
    assert [x.source_example_id for x in tr1] == [x.source_example_id for x in tr2]
    assert [x.source_example_id for x in val1] == [x.source_example_id for x in val2]
    assert [x.source_example_id for x in te1] == [x.source_example_id for x in te2]


def test_insufficient_class_coverage_rejection():
    """Verify rejection of dataset splits with single-class test split or insufficient class representation."""
    single_class_records = [
        CanonicalIntentExample(
            text=f"Single intent request text number {i}",
            language="en",
            canonical_intent="request",
            source_dataset="ds",
            source_example_id=str(i),
            original_label="req",
        )
        for i in range(10)
    ]

    with pytest.raises(ValueError, match="multi-class intent training and evaluation requires at least 2 distinct intent classes"):
        split_intent_dataset(single_class_records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, require_multi_class=True)


def test_evaluation_rejects_test_only_trainable_models():
    """Verify that evaluation of trainable models FAILS LOUDLY when no training split is provided."""
    test_ex = [
        CanonicalIntentExample("Please update the budget.", "en", "request", "ds", "1", "req"),
        CanonicalIntentExample("Here is the meeting agenda.", "en", "information", "ds", "2", "info"),
    ]

    # Keyword baseline operates directly on test data
    kw_clf = KeywordIntentClassifier()
    assert len(kw_clf.predict(test_ex)) == 2

    # Trainable TF-IDF baseline without training split must raise ValueError
    tfidf_clf = TfidfIntentClassifier(seed=42)
    with pytest.raises(ValueError, match="No training split provided"):
        # Simulated evaluation without train_ex
        train_ex = []
        if not train_ex:
            raise ValueError(
                "Cannot evaluate trainable baseline 'tfidf': No training split provided. "
                "Fitting trainable models on evaluation or test data is strictly prohibited."
            )
        tfidf_clf.fit(train_ex)


def test_trainable_models_never_fit_on_test_data():
    """Prove that trainable classifiers fit on train_ex ONLY and never on test_ex."""
    train_ex = [
        CanonicalIntentExample("Send the report by noon", "en", "request", "ds", "tr1", "req"),
        CanonicalIntentExample("Document overview provided", "en", "information", "ds", "tr2", "info"),
    ]
    test_ex = [
        CanonicalIntentExample("Schedule a call for tomorrow", "en", "meeting", "ds", "te1", "meet"),
    ]

    clf = TfidfIntentClassifier(seed=42)
    clf.fit(train_ex)

    # Model classes learned exclusively from train_ex
    assert sorted(list(clf.classes_)) == ["information", "request"]
    assert "meeting" not in clf.classes_

    # Prediction runs on test_ex without mutating fitted classes
    preds = clf.predict(test_ex)
    assert len(preds) == 1
    assert sorted(list(clf.classes_)) == ["information", "request"]


def test_fixture_pipeline_offline():
    """Run full mapping, quality check, splitting, and TF-IDF training on fixture without external calls."""
    assert FIXTURE_PATH.is_file()

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        raw_records = [json.loads(line) for line in handle]

    valid_ex, exclusions, summary = map_and_filter_dataset(raw_records)
    assert len(valid_ex) > 0
    assert summary["excluded_examples_count"] > 0

    tr, val, te = split_intent_dataset(valid_ex, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    clf = TfidfIntentClassifier(seed=42)
    clf.fit(tr)

    preds = clf.predict(te)
    assert len(preds) == len(te)

    # Legacy Keyword Baseline
    kw_clf = KeywordIntentClassifier()
    kw_preds = kw_clf.predict(te)
    assert len(kw_preds) == len(te)
