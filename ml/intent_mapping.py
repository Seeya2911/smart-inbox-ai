"""Defensible Intent Label Mapping and Exclusion Tracking Module.

Only source labels that map defensibly to the Smart Inbox intent taxonomy are retained.
Labels that cannot be mapped confidently are excluded from supervised training with
their exclusion reasons and counts recorded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ml.schema import ALLOWED_INTENTS, CanonicalIntentExample

# Defensible mappings from source datasets to canonical Smart Inbox intents
ENRON_INTENT_MAPPING: Dict[str, Optional[str]] = {
    "ACTION_REQUIRED": "request",
    "positive": "request",
    # NO_ACTION_REQUIRED is excluded because absence of action requirement does not
    # uniquely specify an intent (it could be information, notification, newsletter, etc.)
    "NO_ACTION_REQUIRED": None,
    "negative": None,
}

MASSIVE_INTENT_MAPPING: Dict[str, Optional[str]] = {
    "email_sendemail": "request",
    "email_addcontact": "request",
    "email_query": "information",
    "email_querycontact": "information",
}


@dataclass(frozen=True)
class ExclusionRecord:
    """Record of an example excluded due to indefensible label mapping."""

    source_example_id: str
    source_dataset: str
    original_label: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_canonical_intent(
    source_dataset: str,
    original_label: str,
    raw_intent: str = "",
    custom_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], str]:
    """Resolve original label to canonical intent or provide exclusion reason.

    Returns:
        (canonical_intent_or_none, exclusion_reason_or_empty_str)
    """
    if custom_mapping and original_label in custom_mapping:
        mapped = custom_mapping[original_label]
        if mapped in ALLOWED_INTENTS:
            return mapped, ""
        return None, f"Custom mapping for '{original_label}' target '{mapped}' is not in ALLOWED_INTENTS"

    # Already canonical
    if original_label.lower() in ALLOWED_INTENTS:
        return original_label.lower(), ""

    dataset_lower = source_dataset.lower()

    if "synthetic" in dataset_lower:
        target = raw_intent.lower() if raw_intent else original_label.lower()
        if target in ALLOWED_INTENTS:
            return target, ""
        return None, f"Synthetic intent label '{target}' is not in ALLOWED_INTENTS"

    if "enron" in dataset_lower:
        if original_label in ENRON_INTENT_MAPPING:
            target = ENRON_INTENT_MAPPING[original_label]
            if target is not None:
                return target, ""
            return None, (
                f"Uncertain mapping: Enron label '{original_label}' does not map confidently to a single "
                "Smart Inbox intent class (could be information, notification, promotion, etc.)"
            )

    if "massive" in dataset_lower:
        if original_label in MASSIVE_INTENT_MAPPING:
            target = MASSIVE_INTENT_MAPPING[original_label]
            if target is not None:
                return target, ""
        return None, f"Uncertain mapping: MASSIVE intent '{original_label}' is outside email intent taxonomy"

    return None, f"No defensible intent mapping defined for dataset '{source_dataset}' label '{original_label}'"


def map_raw_record(
    raw_record: Dict[str, Any],
    custom_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[CanonicalIntentExample], Optional[ExclusionRecord]]:
    """Map a single raw record to a CanonicalIntentExample or ExclusionRecord."""
    source_dataset = str(raw_record.get("source_dataset", "unknown")).strip()
    source_id = str(raw_record.get("source_example_id", raw_record.get("id", raw_record.get("source_id", "")))).strip()
    original_label = str(
        raw_record.get("original_label", raw_record.get("action_intent", raw_record.get("intent", raw_record.get("source_intent", ""))))
    ).strip()
    raw_intent = str(raw_record.get("intent", raw_record.get("canonical_intent", ""))).strip()

    canonical_intent, exclusion_reason = resolve_canonical_intent(
        source_dataset=source_dataset,
        original_label=original_label,
        raw_intent=raw_intent,
        custom_mapping=custom_mapping,
    )

    if canonical_intent is None:
        exclusion = ExclusionRecord(
            source_example_id=source_id,
            source_dataset=source_dataset,
            original_label=original_label,
            reason=exclusion_reason,
        )
        return None, exclusion

    # Create record dict with mapped canonical intent
    record_payload = dict(raw_record)
    record_payload["canonical_intent"] = canonical_intent
    record_payload["original_label"] = original_label
    record_payload["source_dataset"] = source_dataset
    record_payload["source_example_id"] = source_id

    example = CanonicalIntentExample.from_dict(record_payload)
    return example, None


def map_and_filter_dataset(
    records: List[Dict[str, Any]],
    custom_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[List[CanonicalIntentExample], List[ExclusionRecord], Dict[str, Any]]:
    """Map a dataset of raw records, filtering unmappable records and collecting exclusion reports."""
    valid_examples: List[CanonicalIntentExample] = []
    exclusions: List[ExclusionRecord] = []
    exclusion_reasons: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}

    for record in records:
        example, exclusion = map_raw_record(record, custom_mapping)
        if example is not None:
            valid_examples.append(example)
            class_counts[example.canonical_intent] = class_counts.get(example.canonical_intent, 0) + 1
        elif exclusion is not None:
            exclusions.append(exclusion)
            exclusion_reasons[exclusion.reason] = exclusion_reasons.get(exclusion.reason, 0) + 1

    summary = {
        "total_input_examples": len(records),
        "mapped_examples_count": len(valid_examples),
        "excluded_examples_count": len(exclusions),
        "exclusion_reasons": exclusion_reasons,
        "mapped_class_distribution": class_counts,
    }

    return valid_examples, exclusions, summary
