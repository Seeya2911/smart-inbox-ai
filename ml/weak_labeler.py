"""Dual Weak Labeler & 3-Population Score Router.

Wraps PriorityTagger and IntentRuleEngine to assign weak priority and intent labels,
calculates uncalibrated `rule_score`, and routes emails into High Rule-Score candidates,
Ambiguous review queue, and Low-Signal pool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ml.intent_rules import IntentRuleEngine
from ml.schema import CanonicalEmailExample
from priority_tagging import PriorityTagger


class DualWeakLabeler:
    """Combines priority tagging rules and intent rules for weak supervision."""

    def __init__(
        self,
        high_threshold: float = 3.0,
        low_threshold: float = 1.0,
    ) -> None:
        self.priority_tagger = PriorityTagger()
        self.intent_engine = IntentRuleEngine()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate_email(self, data: Dict[str, Any]) -> CanonicalEmailExample:
        """Evaluate raw email record and assign weak labels, rule_score, and routing."""
        subject = str(data.get("subject", "")).strip()
        body = str(data.get("body", data.get("text", ""))).strip()
        sender = str(data.get("sender", "")).strip()

        # Priority tagger output
        pri_res = self.priority_tagger.tag_email(
            {"id": data.get("id", "0"), "subject": subject, "body": body, "sender": sender}
        )

        p_tag = pri_res.get("tag", "GENERAL")
        # Map PriorityTagger tag to Canonical Priority
        if p_tag in {"URGENT", "FINANCIAL", "SECURITY"}:
            canonical_priority = "high"
        elif p_tag in {"MEETING", "IMPORTANT"}:
            canonical_priority = "medium"
        else:
            canonical_priority = "low"

        reasons = pri_res.get("reasoning", [])
        pri_score = max(pri_res.get("all_scores", {}).values()) if pri_res.get("all_scores") else 0.0

        # Intent rule engine output
        weak_intent, intent_score, intent_reasons = self.intent_engine.predict_weak_intent(subject, body)
        reasons.extend(intent_reasons)

        total_rule_score = round(pri_score + intent_score, 3)

        payload = dict(data)
        payload["subject"] = subject
        payload["body"] = body
        payload["intent"] = weak_intent
        payload["priority"] = canonical_priority
        payload["priority_reasons"] = reasons
        payload["label_source"] = "rules"
        payload["rule_score"] = total_rule_score
        payload["label_confidence"] = min(1.0, total_rule_score / 10.0)

        return CanonicalEmailExample.from_dict(payload)

    def route_populations(
        self,
        examples: List[CanonicalEmailExample],
    ) -> Tuple[List[CanonicalEmailExample], List[CanonicalEmailExample], List[CanonicalEmailExample]]:
        """Route emails into (high_score_candidates, ambiguous_review_queue, low_signal_pool)."""
        high_score = []
        ambiguous = []
        low_signal = []

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
    """Process a JSON/JSONL dataset file through weak labeler and export routed artifact."""
    labeler = DualWeakLabeler(high_threshold, low_threshold)
    records = []

    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        records = json.loads(input_path.read_text(encoding="utf-8"))

    evaluated = [labeler.evaluate_email(r) for r in records]
    high, amb, low = labeler.route_populations(evaluated)

    summary = {
        "total_records": len(evaluated),
        "high_score_candidates": len(high),
        "ambiguous_review_queue": len(amb),
        "low_signal_pool": len(low),
        "output_file": str(output_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for e in evaluated:
            f.write(json.dumps(e.to_dict()) + "\n")

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dual Weak Labeler & Score Router CLI")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL file path")
    parser.add_argument("--output", type=str, default="artifacts/weak_labeled.jsonl", help="Output JSON path")
    parser.add_argument("--high-threshold", type=float, default=3.0, help="High rule_score threshold")
    parser.add_argument("--low-threshold", type=float, default=1.0, help="Low rule_score threshold")
    args = parser.parse_args()

    summary = process_dataset_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
