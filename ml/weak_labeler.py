"""Dual Weak Labeler & 3-Population Score Router.

Rules are weak supervision only. Their numeric output is stored as an
uncalibrated ``rule_score`` and is never presented as a probability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ml.intent_rules import IntentRuleEngine
from ml.schema import CanonicalEmailExample
from priority_tagging import PriorityTagger


class DualWeakLabeler:
    """Combine deterministic priority and intent rules for weak supervision."""

    def __init__(
        self,
        high_threshold: float = 3.0,
        low_threshold: float = 1.0,
    ) -> None:
        # Disable persisted user feedback here. Weak-label generation must be
        # reproducible and must not change because the application learned from
        # a user's private mailbox.
        self.priority_tagger = PriorityTagger(feedback_file="", confidence_file="")
        self.intent_engine = IntentRuleEngine()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate_email(self, data: Dict[str, Any]) -> CanonicalEmailExample:
        """Assign weak labels and an uncalibrated rule score to one email."""
        subject = str(data.get("subject", "")).strip()
        body = str(data.get("body", data.get("text", ""))).strip()
        sender = str(data.get("sender", "")).strip()
        if not (subject or body):
            raise ValueError("Cannot weak-label an email with empty subject and body")

        pri_res = self.priority_tagger.tag_email(
            {"id": data.get("id", "0"), "subject": subject, "body": body, "sender": sender}
        )

        p_tag = pri_res.get("tag", "GENERAL")
        if p_tag in {"URGENT", "FINANCIAL", "SECURITY"}:
            canonical_priority = "high"
        elif p_tag in {"MEETING", "IMPORTANT"}:
            canonical_priority = "medium"
        else:
            canonical_priority = "low"

        reasons = list(pri_res.get("reasoning", []))
        pri_score = max(pri_res.get("all_scores", {}).values()) if pri_res.get("all_scores") else 0.0

        weak_intent, intent_score, intent_reasons = self.intent_engine.predict_weak_intent(subject, body)
        reasons.extend(intent_reasons)
        total_rule_score = round(pri_score + intent_score, 3)

        payload = dict(data)
        payload.update(
            {
                "subject": subject,
                "body": body,
                "intent": weak_intent,
                "priority": canonical_priority,
                "priority_reasons": reasons,
                "label_source": "rules",
                "rule_score": total_rule_score,
                # This is deliberately NOT total_rule_score / 10. Rules are
                # uncalibrated; confidence stays unset until calibration.
                "label_confidence": 0.0,
            }
        )
        return CanonicalEmailExample.from_dict(payload)

    def route_populations(
        self,
        examples: List[CanonicalEmailExample],
    ) -> Tuple[List[CanonicalEmailExample], List[CanonicalEmailExample], List[CanonicalEmailExample]]:
        """Return (high-score candidates, ambiguous review, low-signal pool)."""
        high_score: List[CanonicalEmailExample] = []
        ambiguous: List[CanonicalEmailExample] = []
        low_signal: List[CanonicalEmailExample] = []
        for ex in examples:
            if ex.rule_score >= self.high_threshold:
                high_score.append(ex)
            elif ex.rule_score >= self.low_threshold:
                ambiguous.append(ex)
            else:
                low_signal.append(ex)
        return high_score, ambiguous, low_signal


def process_dataset_file(
    input_path: Path,
    output_path: Path,
    high_threshold: float = 3.0,
    low_threshold: float = 1.0,
) -> Dict[str, Any]:
    """Process a JSON/JSONL dataset and export weak labels plus routing metadata."""
    labeler = DualWeakLabeler(high_threshold, low_threshold)
    records: List[Dict[str, Any]] = []
    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    else:
        records = json.loads(input_path.read_text(encoding="utf-8"))

    evaluated: List[CanonicalEmailExample] = []
    rejected = 0
    for record in records:
        try:
            evaluated.append(labeler.evaluate_email(record))
        except ValueError:
            rejected += 1

    high, ambiguous, low = labeler.route_populations(evaluated)
    summary = {
        "total_records": len(records),
        "labeled_records": len(evaluated),
        "rejected_records": rejected,
        "high_score_candidates": len(high),
        "ambiguous_review_queue": len(ambiguous),
        "low_signal_pool": len(low),
        "output_file": str(output_path),
        "label_confidence_policy": "0.0 until calibrated against human-reviewed labels",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in evaluated:
            handle.write(json.dumps(example.to_dict()) + "\n")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dual Weak Labeler & Score Router CLI")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL file path")
    parser.add_argument("--output", type=str, default="artifacts/weak_labeled.jsonl")
    parser.add_argument("--high-threshold", type=float, default=3.0)
    parser.add_argument("--low-threshold", type=float, default=1.0)
    args = parser.parse_args()

    print(json.dumps(process_dataset_file(Path(args.input), Path(args.output), args.high_threshold, args.low_threshold), indent=2))


if __name__ == "__main__":
    main()
