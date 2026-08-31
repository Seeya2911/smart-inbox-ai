"""Tests for ml.v2_dataset_loader — XLSX loading, audit, and leakage safety.

CRITICAL: At least one test proves that intent_reason, priority_reason,
label_confidence, source, label_source, and is_synthetic are NOT included
in the model input text (full_text).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.schema import ALLOWED_INTENTS, ALLOWED_PRIORITIES, CanonicalEmailExample
from ml.v2_dataset_loader import (
    _ColumnMap,
    _normalize,
    _row_to_example,
    export_to_jsonl,
    load_from_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_example(**overrides) -> CanonicalEmailExample:
    defaults = dict(
        id="synthetic_001",
        subject="Test subject",
        body="This is the body of the email.",
        intent="request",
        priority="high",
        source="synthetic",
        label_source="llm",
        label_confidence=0.9,
        is_synthetic=True,
    )
    defaults.update(overrides)
    return CanonicalEmailExample.from_dict(defaults)


# ---------------------------------------------------------------------------
# Leakage safety tests — THE MOST CRITICAL
# ---------------------------------------------------------------------------

class TestModelFeatureLeakage:
    """Prove that metadata fields never appear in model input text."""

    def test_intent_reason_not_in_full_text(self):
        ex = CanonicalEmailExample(
            id="synthetic_001",
            subject="Meeting tomorrow",
            body="Let's schedule a call.",
            intent="meeting",
            priority="medium",
            source="synthetic",
            label_source="llm",
            label_confidence=0.9,
            is_synthetic=True,
            llm_intent_reason="Because the email discusses scheduling a meeting",
            llm_priority_reason="Meeting is tomorrow so medium urgency",
        )
        full_text = ex.full_text
        assert "Because the email discusses scheduling a meeting" not in full_text, (
            "intent_reason MUST NOT appear in full_text / model features"
        )
        assert "medium urgency" not in full_text, (
            "priority_reason MUST NOT appear in full_text / model features"
        )

    def test_priority_reason_not_in_full_text(self):
        ex = CanonicalEmailExample(
            id="synthetic_002",
            subject="Payment failed",
            body="Your payment was declined.",
            intent="transactional",
            priority="high",
            source="synthetic",
            label_source="llm",
            label_confidence=0.85,
            is_synthetic=True,
            llm_priority_reason="Financial impact warrants high priority",
        )
        assert "Financial impact warrants high priority" not in ex.full_text

    def test_label_confidence_not_in_full_text(self):
        ex = _make_example(label_confidence=0.76543)
        assert "0.76543" not in ex.full_text
        assert "label_confidence" not in ex.full_text

    def test_source_not_in_full_text(self):
        """Source is metadata, not a feature."""
        ex = _make_example(source="synthetic", body="Email body about an invoice.")
        assert "synthetic" not in ex.full_text or "synthetic" in ex.subject or "synthetic" in ex.body
        # More precisely: the FIELD name 'source' must not be there
        assert "source" not in ex.full_text

    def test_is_synthetic_not_in_full_text(self):
        ex = _make_example(is_synthetic=True)
        assert "is_synthetic" not in ex.full_text
        assert "True" not in ex.full_text or "True" in ex.subject or "True" in ex.body

    def test_label_source_not_in_full_text(self):
        ex = _make_example(label_source="llm")
        assert "label_source" not in ex.full_text

    def test_id_not_in_full_text(self):
        ex = CanonicalEmailExample(
            id="synthetic_UNIQUE_ID_12345",
            subject="Hello",
            body="Body here.",
            intent="information",
            priority="low",
            source="synthetic",
            label_source="llm",
            label_confidence=1.0,
            is_synthetic=True,
        )
        assert "UNIQUE_ID_12345" not in ex.full_text

    def test_full_text_contains_only_subject_and_body(self):
        ex = CanonicalEmailExample(
            id="synthetic_009",
            subject="Invoice reminder",
            body="Please pay by Friday.",
            intent="transactional",
            priority="medium",
            source="synthetic",
            label_source="llm",
            label_confidence=0.9,
            is_synthetic=True,
        )
        assert "Invoice reminder" in ex.full_text
        assert "Please pay by Friday" in ex.full_text
        # Check format
        assert ex.full_text.startswith("Subject: Invoice reminder")


# ---------------------------------------------------------------------------
# Column map tests
# ---------------------------------------------------------------------------

class TestColumnMap:
    def test_gets_first_matching_alias(self):
        headers = ["index", "body", "intent", "priority", "subject"]
        cm = _ColumnMap(headers)
        from ml.v2_dataset_loader import _BODY_COLS, _SUBJECT_COLS
        assert cm.get(["idx", "body text", "planning", "high", "hello"], _BODY_COLS) == "body text"
        assert cm.get(["idx", "body text", "planning", "high", "hello"], _SUBJECT_COLS) == "hello"

    def test_returns_default_when_alias_missing(self):
        headers = ["body"]
        cm = _ColumnMap(headers)
        from ml.v2_dataset_loader import _SOURCE_COLS
        val = cm.get(["email body"], _SOURCE_COLS, default="synthetic")
        assert val == "synthetic"


# ---------------------------------------------------------------------------
# Row → example conversion tests
# ---------------------------------------------------------------------------

class TestRowToExample:
    def _make_col_map(self):
        headers = [
            "index", "id", "subject", "body", "intent", "priority",
            "intent_reason", "priority_reason", "label_confidence",
            "source", "label_source", "is_synthetic",
        ]
        return _ColumnMap(headers), headers

    def _make_row(self, **overrides):
        base = {
            "index": 1,
            "id": "row_1",
            "subject": "Test Subject",
            "body": "Test body content for the email.",
            "intent": "information",
            "priority": "low",
            "intent_reason": "Provides information",
            "priority_reason": "Low urgency content",
            "label_confidence": 0.9,
            "source": "synthetic",
            "label_source": "llm",
            "is_synthetic": True,
        }
        base.update(overrides)
        cm, headers = self._make_col_map()
        return [base[h] for h in headers], cm

    def test_valid_row_creates_example(self):
        row, cm = self._make_row()
        ex, reason = _row_to_example(row, 2, cm)
        assert ex is not None
        assert reason is None
        assert ex.intent == "information"
        assert ex.priority == "low"

    def test_empty_body_is_rejected(self):
        row, cm = self._make_row(body="")
        ex, reason = _row_to_example(row, 2, cm)
        assert ex is None
        assert "empty body" in (reason or "").lower()

    def test_invalid_intent_is_rejected(self):
        row, cm = self._make_row(intent="spam")
        ex, reason = _row_to_example(row, 2, cm)
        assert ex is None
        assert "invalid intent" in (reason or "").lower()

    def test_invalid_priority_is_rejected(self):
        row, cm = self._make_row(priority="critical")
        ex, reason = _row_to_example(row, 2, cm)
        assert ex is None
        assert "invalid priority" in (reason or "").lower()

    def test_all_11_intents_accepted(self):
        for intent in ALLOWED_INTENTS:
            row, cm = self._make_row(intent=intent)
            ex, reason = _row_to_example(row, 2, cm)
            assert ex is not None, f"Intent '{intent}' should be accepted but was rejected: {reason}"

    def test_all_3_priorities_accepted(self):
        for priority in ALLOWED_PRIORITIES:
            row, cm = self._make_row(priority=priority)
            ex, reason = _row_to_example(row, 2, cm)
            assert ex is not None, f"Priority '{priority}' should be accepted but was rejected: {reason}"

    def test_intent_reason_stored_in_metadata_not_text(self):
        row, cm = self._make_row(intent_reason="SPECIAL_MARKER_XYZ")
        ex, reason = _row_to_example(row, 2, cm)
        assert ex is not None
        assert "SPECIAL_MARKER_XYZ" not in ex.full_text
        assert ex.llm_intent_reason == "SPECIAL_MARKER_XYZ"

    def test_priority_reason_stored_in_metadata_not_text(self):
        row, cm = self._make_row(priority_reason="PRIORITY_REASON_UNIQUE_99")
        ex, reason = _row_to_example(row, 2, cm)
        assert ex is not None
        assert "PRIORITY_REASON_UNIQUE_99" not in ex.full_text
        assert ex.llm_priority_reason == "PRIORITY_REASON_UNIQUE_99"


# ---------------------------------------------------------------------------
# JSONL round-trip tests
# ---------------------------------------------------------------------------

class TestJsonlRoundTrip:
    def test_export_and_reload(self, tmp_path: Path):
        examples = [
            _make_example(id=f"synthetic_{i:03d}", body=f"Body for email {i}", intent="request")
            for i in range(5)
        ]
        out = tmp_path / "test.jsonl"
        export_to_jsonl(examples, out)
        reloaded = load_from_jsonl(out)
        assert len(reloaded) == 5
        for orig, rel in zip(examples, reloaded):
            assert orig.id == rel.id
            assert orig.intent == rel.intent
            assert orig.priority == rel.priority

    def test_all_fields_preserved_in_jsonl(self, tmp_path: Path):
        ex = CanonicalEmailExample(
            id="synthetic_001",
            subject="Test Subject",
            body="Test body.",
            intent="security",
            priority="high",
            source="synthetic",
            label_source="llm",
            label_confidence=0.88,
            is_synthetic=True,
            llm_intent_reason="Security threat detected",
            llm_priority_reason="High urgency security issue",
        )
        out = tmp_path / "single.jsonl"
        export_to_jsonl([ex], out)
        reloaded = load_from_jsonl(out)
        assert len(reloaded) == 1
        r = reloaded[0]
        assert r.llm_intent_reason == "Security threat detected"
        assert r.llm_priority_reason == "High urgency security issue"
        assert r.label_confidence == pytest.approx(0.88, abs=1e-3)


# ---------------------------------------------------------------------------
# Label validation coverage
# ---------------------------------------------------------------------------

class TestLabelValidation:
    def test_security_intent_is_valid(self):
        ex = _make_example(intent="security")
        assert ex.intent == "security"

    def test_transactional_intent_is_valid(self):
        ex = _make_example(intent="transactional")
        assert ex.intent == "transactional"

    def test_follow_up_intent_is_valid(self):
        ex = _make_example(intent="follow_up")
        assert ex.intent == "follow_up"

    def test_high_priority_is_valid(self):
        ex = _make_example(priority="high")
        assert ex.priority == "high"

    def test_medium_priority_is_valid(self):
        ex = _make_example(priority="medium")
        assert ex.priority == "medium"

    def test_exactly_11_intents_in_schema(self):
        assert len(ALLOWED_INTENTS) == 11, (
            f"Expected 11 intents, got {len(ALLOWED_INTENTS)}: {sorted(ALLOWED_INTENTS)}"
        )
        assert "security" in ALLOWED_INTENTS
        assert "transactional" in ALLOWED_INTENTS

    def test_exactly_3_priorities_in_schema(self):
        assert len(ALLOWED_PRIORITIES) == 3
        assert "high" in ALLOWED_PRIORITIES
        assert "medium" in ALLOWED_PRIORITIES
        assert "low" in ALLOWED_PRIORITIES
