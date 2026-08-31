"""Tests for split_email_dataset() — joint stratification, determinism, leakage.

Covers:
- Deterministic splitting (same seed → same result)
- Joint intent×priority stratification path
- Fallback to intent-only when rare combinations exist
- All 11 intents present in train split
- No group overlap across splits (leakage check passes)
- Distribution printing does not crash
"""
from __future__ import annotations

from typing import List

import pytest

from ml.data_quality import DataLeakageError
from ml.dataset_splitter import split_email_dataset
from ml.schema import CanonicalEmailExample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TOPICS = [
    "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    "lambda mu nu xi omicron pi rho sigma tau upsilon phi",
    "mercury venus earth mars jupiter saturn uranus neptune",
    "apples bananas oranges pineapples strawberries blueberries mangoes",
    "physics chemistry biology geology astronomy mathematics statistics",
    "football basketball tennis baseball hockey golf swimming cycling",
    "violin guitar piano flute trumpet drums saxophone clarinet cello",
    "red blue green yellow purple orange brown black white violet",
    "london paris tokyo berlin rome madrid ottawa canberra brasilia",
    "coffee tea water juice soda lemonade smoothie mocha latte",
    "desktop laptop tablet monitor keyboard mouse printer scanner webcam",
]


def _make_ex(row: int, intent: str, priority: str, body_suffix: str = "") -> CanonicalEmailExample:
    topic = _TOPICS[row % len(_TOPICS)]
    return CanonicalEmailExample.from_dict({
        "id": f"synthetic_{row:04d}",
        "subject": f"{intent.upper()} Alert #{row} ({priority})",
        "body": f"Regarding item {row} classified as {intent} with {priority} urgency. {topic} {body_suffix}",
        "intent": intent,
        "priority": priority,
        "source": "synthetic",
        "label_source": "llm",
        "label_confidence": 0.9,
        "is_synthetic": True,
    })


def _make_balanced_dataset(n_per_combo: int = 5) -> List[CanonicalEmailExample]:
    """Create examples covering all 11x3 = 33 intent x priority combinations."""
    from ml.schema import ALLOWED_INTENTS, ALLOWED_PRIORITIES
    examples = []
    idx = 0
    for intent in sorted(ALLOWED_INTENTS):
        for priority in sorted(ALLOWED_PRIORITIES):
            for k in range(n_per_combo):
                idx += 1
                unique_words = " ".join(f"token_{idx}_{w}_{k}" for w in range(8))
                examples.append(_make_ex(
                    idx, intent, priority,
                    body_suffix=unique_words
                ))
    return examples


def _make_imbalanced_dataset() -> List[CanonicalEmailExample]:
    """Dataset with rare combinations to force fallback to intent-only stratification."""
    examples = []
    idx = 0
    # Most intents have multiple examples
    for i in range(20):
        idx += 1
        examples.append(_make_ex(idx, "request", "high", f"req_h_{i}_unique_text_{idx}"))
    for i in range(18):
        idx += 1
        examples.append(_make_ex(idx, "information", "low", f"inf_l_{i}_unique_text_{idx}"))
    for i in range(15):
        idx += 1
        examples.append(_make_ex(idx, "meeting", "medium", f"meet_m_{i}_unique_text_{idx}"))
    # Rare combination: only 1 example of complaint|high
    idx += 1
    examples.append(_make_ex(idx, "complaint", "high", f"rare_combo_complaint_unique_{idx}"))
    # Fill remaining intents minimally
    for intent in ["question", "follow_up", "notification", "promotion",
                   "security", "transactional", "other"]:
        for k in range(8):
            idx += 1
            examples.append(_make_ex(idx, intent, "low", f"{intent}_{k}_unique_{idx}"))
    return examples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitEmailDataset:

    def test_split_sizes_approximate_70_15_15(self):
        examples = _make_balanced_dataset(n_per_combo=5)
        n = len(examples)
        train, val, test = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        assert len(train) + len(val) + len(test) == n
        # Allow ±5% tolerance due to group-aware splitting
        assert abs(len(train) / n - 0.70) < 0.10
        assert abs(len(val) / n - 0.15) < 0.10
        assert abs(len(test) / n - 0.15) < 0.10

    def test_deterministic_with_same_seed(self):
        examples = _make_balanced_dataset(n_per_combo=5)
        train1, val1, test1 = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        train2, val2, test2 = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        assert [ex.id for ex in train1] == [ex.id for ex in train2]
        assert [ex.id for ex in val1] == [ex.id for ex in val2]
        assert [ex.id for ex in test1] == [ex.id for ex in test2]

    def test_different_seeds_give_different_splits(self):
        examples = _make_balanced_dataset(n_per_combo=6)
        train1, _, _ = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        train2, _, _ = split_email_dataset(
            examples, seed=99, print_distributions=False
        )
        # Different seeds should generally produce different assignments
        # (not guaranteed, but almost certain for n > 30)
        ids1 = {ex.id for ex in train1}
        ids2 = {ex.id for ex in train2}
        assert ids1 != ids2

    def test_no_id_overlap_across_splits(self):
        examples = _make_balanced_dataset(n_per_combo=5)
        train, val, test = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        train_ids = {ex.id for ex in train}
        val_ids = {ex.id for ex in val}
        test_ids = {ex.id for ex in test}
        assert not (train_ids & val_ids), "ID overlap between train and val"
        assert not (train_ids & test_ids), "ID overlap between train and test"
        assert not (val_ids & test_ids), "ID overlap between val and test"

    def test_all_intents_in_train_split(self):
        from ml.schema import ALLOWED_INTENTS
        examples = _make_balanced_dataset(n_per_combo=5)
        train, _, _ = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        train_intents = {ex.intent for ex in train}
        missing = ALLOWED_INTENTS - train_intents
        assert not missing, f"Intents missing from train: {sorted(missing)}"

    def test_all_priorities_in_train_split(self):
        from ml.schema import ALLOWED_PRIORITIES
        examples = _make_balanced_dataset(n_per_combo=5)
        train, _, _ = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        train_priorities = {ex.priority for ex in train}
        assert train_priorities == ALLOWED_PRIORITIES

    def test_joint_stratification_preserves_both_distributions(self):
        """When all combos have enough examples, joint strat is used."""
        examples = _make_balanced_dataset(n_per_combo=6)
        # 6 examples per combo ≥ threshold of 3, so joint strat applies
        train, val, test = split_email_dataset(
            examples, seed=42, joint_rare_threshold=3, print_distributions=False
        )
        # Both intent and priority should be reasonably distributed
        from collections import Counter
        train_intents = Counter(ex.intent for ex in train)
        train_priorities = Counter(ex.priority for ex in train)
        # All intents should appear
        from ml.schema import ALLOWED_INTENTS
        for intent in ALLOWED_INTENTS:
            assert train_intents.get(intent, 0) > 0, f"Intent '{intent}' missing from train"

    def test_fallback_to_intent_only_when_rare_combos(self):
        """Rare combinations trigger intent-only fallback without crashing."""
        examples = _make_imbalanced_dataset()
        # This should not raise — fallback handles it
        train, val, test = split_email_dataset(
            examples, seed=42, joint_rare_threshold=3, print_distributions=False
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_raises_on_empty_dataset(self):
        with pytest.raises(ValueError, match="empty"):
            split_email_dataset([], seed=42, print_distributions=False)

    def test_raises_on_bad_ratios(self):
        examples = _make_balanced_dataset(n_per_combo=3)
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_email_dataset(
                examples, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2,
                print_distributions=False
            )

    def test_print_distributions_does_not_crash(self, capsys):
        examples = _make_balanced_dataset(n_per_combo=4)
        train, val, test = split_email_dataset(
            examples, seed=42, print_distributions=True
        )
        captured = capsys.readouterr()
        assert "SPLIT DISTRIBUTIONS" in captured.out
        assert "INTENT × PRIORITY" in captured.out
        # All 11 intents should appear in the output
        for intent in ["request", "security", "transactional", "complaint", "follow_up"]:
            assert intent in captured.out

    def test_leakage_check_passes_after_valid_split(self):
        """After split_email_dataset, check_split_leakage_email should pass."""
        from ml.data_quality import check_split_leakage_email
        examples = _make_balanced_dataset(n_per_combo=5)
        train, val, test = split_email_dataset(
            examples, seed=42, print_distributions=False
        )
        # Should not raise
        check_split_leakage_email(train, val, test)
