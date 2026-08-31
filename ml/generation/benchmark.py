"""Benchmark suite for local pretrained generative foundation (FLAN-T5-base).

Separately evaluates:
1. RAW MODEL OUTPUT (before parsing or normalization)
2. PARSED/VALIDATED OUTPUT (after schema validation and safe nulling)

IMPORTANT: Uses a dedicated evaluation subset from the validation pool.
Does NOT use or contaminate the frozen classification test set.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ml.deduplication import strip_email_boilerplate
from ml.generation.inference import process_email
from ml.generation.model import load_model
from ml.generation.schemas import ALLOWED_ACTION_TYPES, GenerationOutput, SuggestedAction
from ml.schema import CanonicalEmailExample


def extract_key_entities(text: str) -> List[str]:
    """Find dates, monetary amounts, and numbers to check preservation in summary."""
    patterns = [
        r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
        r"\b\d+%",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    ]
    matches = []
    for pat in patterns:
        matches.extend(re.findall(pat, text, re.IGNORECASE))
    return list(set(matches))


def get_ground_truth_action_type(ex: CanonicalEmailExample) -> str:
    """Determine expected ground-truth action type from email characteristics."""
    intent = str(ex.intent or "").lower()
    subject = str(ex.subject or "").lower()
    body = str(ex.body or "").lower()

    if intent == "meeting" or "meeting" in subject or "schedule" in subject:
        return "create_calendar_event"
    elif intent == "question" or "reply" in subject or "can you" in body:
        return "reply"
    elif intent in ["promotion", "other", "notification"]:
        return "none"
    elif intent == "follow_up":
        return "follow_up"
    elif intent in ["request", "security", "transactional"]:
        if any(k in body for k in ["submit", "reset", "verify", "pay", "send", "approve"]):
            return "create_task"
        return "reply"
    return "none"


def compute_none_metrics(y_true_is_none: List[bool], y_pred_is_none: List[bool]) -> Dict[str, float]:
    """Compute Precision, Recall, and F1 for detecting 'none' (non-actionable) emails."""
    tp = sum(1 for yt, yp in zip(y_true_is_none, y_pred_is_none) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true_is_none, y_pred_is_none) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true_is_none, y_pred_is_none) if yt and not yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_generation_benchmark(
    eval_examples: List[CanonicalEmailExample],
    output_dir: Path,
    max_samples: int = 50,
) -> Dict[str, Any]:
    """Execute generation benchmark with strict separation between Raw Model and Parsed Outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = eval_examples[:max_samples]

    print(f"\n==============================================================")
    print(f"  RUNNING GENERATION BENCHMARK (n={len(samples)} emails)")
    print(f"==============================================================")

    model = load_model()

    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    compression_ratios: List[float] = []
    entity_preservation_scores: List[float] = []

    # Ground truth and evaluation lists
    gt_action_types: List[str] = []
    raw_action_types: List[str] = []
    parsed_action_types: List[str] = []
    raw_is_valid_json: List[bool] = []
    parsed_is_valid_json: List[bool] = []

    t_start = time.time()
    for i, ex in enumerate(samples):
        out = process_email(
            subject=ex.subject,
            body=ex.body,
            intent=ex.intent,
            priority=ex.priority,
            model=model,
        )

        latencies.append(out.latency_ms)

        clean_body = strip_email_boilerplate(ex.body)
        orig_len = max(len(clean_body), 1)
        summ_len = len(out.summary)
        compression = summ_len / orig_len
        compression_ratios.append(compression)

        # Entity preservation
        entities = extract_key_entities(f"{ex.subject} {clean_body}")
        if entities:
            preserved = sum(1 for e in entities if e.lower() in out.summary.lower())
            entity_preservation_scores.append(preserved / len(entities))

        # Ground truth
        gt_action = get_ground_truth_action_type(ex)
        gt_action_types.append(gt_action)

        # 1. Raw Model Output Analysis
        raw_str = out.raw_model_action.strip()
        raw_type = "invalid/unrecognized"
        for act in ALLOWED_ACTION_TYPES:
            if raw_str.lower().startswith(act) or raw_str.lower() == act:
                raw_type = act
                break
        raw_action_types.append(raw_type)

        # Check raw JSON validity (did the model output strict JSON?)
        try:
            json.loads(raw_str)
            raw_is_valid_json.append(True)
        except Exception:
            raw_is_valid_json.append(False)

        # 2. Parsed / Validated Output Analysis
        parsed_type = out.action.action_type
        parsed_action_types.append(parsed_type)
        parsed_is_valid_json.append(parsed_type in ALLOWED_ACTION_TYPES)

        results.append({
            "id": ex.id,
            "subject": ex.subject[:80],
            "intent": ex.intent,
            "priority": ex.priority,
            "raw_model_summary": out.raw_model_summary,
            "raw_model_action": out.raw_model_action,
            "parsed_summary": out.summary,
            "parsed_action": out.action.to_dict(),
            "gt_action_type": gt_action,
            "raw_action_type": raw_type,
            "parsed_action_type": parsed_type,
            "latency_ms": round(out.latency_ms, 2),
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            print(f"  Processed {i + 1}/{len(samples)} emails (Avg latency: {np.mean(latencies):.1f} ms)...")

    total_time = time.time() - t_start

    # Metrics calculation
    # Accuracy: exact match with ground truth
    raw_acc = sum(1 for gt, r in zip(gt_action_types, raw_action_types) if gt == r) / len(samples)
    parsed_acc = sum(1 for gt, p in zip(gt_action_types, parsed_action_types) if gt == p) / len(samples)

    # None detection metrics
    gt_is_none = [gt == "none" for gt in gt_action_types]
    raw_is_none = [r == "none" for r in raw_action_types]
    parsed_is_none = [p == "none" for p in parsed_action_types]

    raw_none_m = compute_none_metrics(gt_is_none, raw_is_none)
    parsed_none_m = compute_none_metrics(gt_is_none, parsed_is_none)

    # False action rate: predicting an action when GT is 'none'
    gt_none_count = max(sum(gt_is_none), 1)
    raw_false_actions = sum(1 for gt_n, r_n in zip(gt_is_none, raw_is_none) if gt_n and not r_n)
    raw_false_action_rate = raw_false_actions / gt_none_count

    parsed_false_actions = sum(1 for gt_n, p_n in zip(gt_is_none, parsed_is_none) if gt_n and not p_n)
    parsed_false_action_rate = parsed_false_actions / gt_none_count

    raw_json_validity = (sum(raw_is_valid_json) / len(samples)) * 100.0
    parsed_json_validity = (sum(parsed_is_valid_json) / len(samples)) * 100.0

    benchmark_summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": model.model_name,
        "device": model.device,
        "sample_size": len(samples),
        "total_time_seconds": round(total_time, 2),
        "latency_metrics": {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "median_ms": round(float(np.median(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        },
        "summarization_metrics": {
            "avg_summary_length_chars": round(float(np.mean([len(r["parsed_summary"]) for r in results])), 1),
            "avg_compression_ratio": round(float(np.mean(compression_ratios)), 3),
            "entity_preservation_rate": round(float(np.mean(entity_preservation_scores)), 3) if entity_preservation_scores else 1.0,
        },
        "raw_model_evaluation": {
            "raw_action_type_accuracy": round(raw_acc * 100.0, 2),
            "raw_none_precision": raw_none_m["precision"],
            "raw_none_recall": raw_none_m["recall"],
            "raw_none_f1": raw_none_m["f1"],
            "raw_false_action_rate": round(raw_false_action_rate * 100.0, 2),
            "raw_json_validity": round(raw_json_validity, 2),
            "raw_action_distribution": {
                act: sum(1 for r in raw_action_types if r == act)
                for act in sorted(list(ALLOWED_ACTION_TYPES) + ["invalid/unrecognized"])
            },
        },
        "parsed_model_evaluation": {
            "parsed_action_type_accuracy": round(parsed_acc * 100.0, 2),
            "parsed_none_precision": parsed_none_m["precision"],
            "parsed_none_recall": parsed_none_m["recall"],
            "parsed_none_f1": parsed_none_m["f1"],
            "parsed_false_action_rate": round(parsed_false_action_rate * 100.0, 2),
            "parsed_json_validity": round(parsed_json_validity, 2),
            "parsed_action_distribution": {
                act: sum(1 for p in parsed_action_types if p == act)
                for act in sorted(list(ALLOWED_ACTION_TYPES))
            },
        },
        "sample_generations": results[:10],
    }

    # Save JSON report
    json_path = output_dir / "generation_benchmark.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(benchmark_summary, fh, indent=2)
    print(f"\n  [OK] Saved benchmark JSON -> {json_path}")

    # Save Markdown report
    md_content = _generate_markdown_report(benchmark_summary)
    md_path = output_dir / "generation_benchmark.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(md_content)
    print(f"  [OK] Saved benchmark Markdown -> {md_path}")

    return benchmark_summary


def _generate_markdown_report(b: Dict[str, Any]) -> str:
    lm = b["latency_metrics"]
    sm = b["summarization_metrics"]
    raw = b["raw_model_evaluation"]
    parsed = b["parsed_model_evaluation"]

    md = []
    md.append("# Smart Inbox AI -- Local Generative Foundation Benchmark")
    md.append("## Raw Model vs. Post-Processing Honest Evaluation")
    md.append(f"\n- **Model:** `{b['model_name']}` (Device: `{b['device']}`)")
    md.append(f"- **Sample Size:** {b['sample_size']} emails (from validation pool)")
    md.append(f"- **Total Benchmark Time:** {b['total_time_seconds']}s")

    md.append("\n## 1. RAW MODEL VS PARSED OUTPUT COMPARISON")
    md.append("| Metric | RAW MODEL OUTPUT | PARSED / VALIDATED OUTPUT | Delta / Impact |")
    md.append("|---|---|---|---|")
    md.append(f"| **Action Type Accuracy** | **{raw['raw_action_type_accuracy']}%** | **{parsed['parsed_action_type_accuracy']}%** | Strict parser rejects unsupported actions |")
    md.append(f"| **'None' Precision** | **{raw['raw_none_precision']}** | **{parsed['parsed_none_precision']}** | Post-processor maps unrecognized outputs to 'none' |")
    md.append(f"| **'None' Recall** | **{raw['raw_none_recall']}** | **{parsed['parsed_none_recall']}** | Non-actionable email recall |")
    md.append(f"| **'None' F1-Score** | **{raw['raw_none_f1']}** | **{parsed['parsed_none_f1']}** | F1 on non-actionable emails |")
    md.append(f"| **False Action Rate** | **{raw['raw_false_action_rate']}%** | **{parsed['parsed_false_action_rate']}%** | Rate of inventing action on 'none' emails |")
    md.append(f"| **Schema / JSON Validity** | **{raw['raw_json_validity']}%** | **{parsed['parsed_json_validity']}%** | Parser ensures 100% schema compliance |")

    md.append("\n## 2. Latency & Throughput (CPU)")
    md.append(f"- **Mean Latency per Email:** **{lm['mean_ms']} ms**")
    md.append(f"- **Median Latency:** **{lm['median_ms']} ms**")
    md.append(f"- **P95 Latency:** **{lm['p95_ms']} ms**")

    md.append("\n## 3. Summarization Quality")
    md.append(f"- **Average Summary Length:** {sm['avg_summary_length_chars']} chars")
    md.append(f"- **Average Compression Ratio:** {sm['avg_compression_ratio'] * 100:.1f}% of original body length")
    md.append(f"- **Key Entity/Date Preservation Rate:** **{sm['entity_preservation_rate'] * 100:.1f}%**")

    md.append("\n## 4. Raw vs. Parsed Distribution Breakdown")
    md.append("### Raw Model Output Distribution:")
    for act, count in sorted(raw["raw_action_distribution"].items(), key=lambda x: -x[1]):
        if count > 0:
            md.append(f"- `{act}`: {count} cases")

    md.append("\n### Parsed Output Distribution:")
    for act, count in sorted(parsed["parsed_action_distribution"].items(), key=lambda x: -x[1]):
        if count > 0:
            md.append(f"- `{act}`: {count} cases")

    md.append("\n## 5. Sample Generations (Raw vs Parsed)")
    for s in b["sample_generations"][:5]:
        md.append(f"### ID: `{s['id']}` (Intent: `{s['intent']}`, Priority: `{s['priority']}`)")
        md.append(f"- **Subject:** {s['subject']}")
        md.append(f"- **Raw Model Summary:** {s['raw_model_summary']}")
        md.append(f"- **Raw Model Action Output:** `{s['raw_model_action']}`")
        md.append(f"- **Parsed Structured Action:** `{s['parsed_action_type']}` -- *{s['parsed_action'].get('title') or 'N/A'}*")

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLAN-T5 generation benchmark with raw vs parsed evaluation")
    parser.add_argument("--dataset-splits", type=str, default="artifacts/canonical_multi_output_dataset.json")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    splits_path = Path(args.dataset_splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits path {splits_path} not found.")

    with splits_path.open("r", encoding="utf-8") as fh:
        split_data = json.load(fh)["splits"]

    val_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["val"]]
    run_generation_benchmark(val_ex, Path(args.output_dir), max_samples=args.max_samples)


if __name__ == "__main__":
    main()
