"""Benchmark suite for local pretrained generative foundation (FLAN-T5-base).

Evaluates:
- Summarization: conciseness ratio, entity/date preservation, token length
- Action Extraction: structured schema validity, non-action detection, date/time coverage
- Inference Performance: per-email latency on CPU/device

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
    # Dollar amounts, dates, percentages, numbers
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


def run_generation_benchmark(
    eval_examples: List[CanonicalEmailExample],
    output_dir: Path,
    max_samples: int = 120,
) -> Dict[str, Any]:
    """Execute generation benchmark over curated evaluation examples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = eval_examples[:max_samples]

    print(f"\n==============================================================")
    print(f"  RUNNING GENERATION BENCHMARK (n={len(samples)} emails)")
    print(f"==============================================================")

    model = load_model()

    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    compression_ratios: List[float] = []
    valid_schema_count = 0
    non_action_count = 0
    entity_preservation_scores: List[float] = []

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
            score = preserved / len(entities)
            entity_preservation_scores.append(score)

        # Action validity
        if out.action.action_type in ALLOWED_ACTION_TYPES:
            valid_schema_count += 1
        if out.action.action_type == "none":
            non_action_count += 1

        results.append({
            "id": ex.id,
            "subject": ex.subject[:80],
            "intent": ex.intent,
            "priority": ex.priority,
            "summary": out.summary,
            "action": out.action.to_dict(),
            "latency_ms": round(out.latency_ms, 2),
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            print(f"  Processed {i + 1}/{len(samples)} emails (Avg latency: {np.mean(latencies):.1f} ms)...")

    total_time = time.time() - t_start

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
            "avg_summary_length_chars": round(float(np.mean([len(r["summary"]) for r in results])), 1),
            "avg_compression_ratio": round(float(np.mean(compression_ratios)), 3),
            "entity_preservation_rate": round(float(np.mean(entity_preservation_scores)), 3) if entity_preservation_scores else 1.0,
        },
        "action_extraction_metrics": {
            "structured_schema_validity_rate": round((valid_schema_count / len(samples)) * 100.0, 2),
            "non_action_rate": round((non_action_count / len(samples)) * 100.0, 2),
            "action_type_distribution": {
                act: sum(1 for r in results if r["action"]["action_type"] == act)
                for act in ALLOWED_ACTION_TYPES
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
    am = b["action_extraction_metrics"]

    md = []
    md.append("# Smart Inbox AI -- Local Generative Foundation Benchmark")
    md.append(f"\n- **Model:** `{b['model_name']}` (Device: `{b['device']}`)")
    md.append(f"- **Sample Size:** {b['sample_size']} emails (from validation pool)")
    md.append(f"- **Total Benchmark Time:** {b['total_time_seconds']}s")

    md.append("\n## 1. Latency & Throughput (CPU)")
    md.append(f"- **Mean Latency per Email:** **{lm['mean_ms']} ms**")
    md.append(f"- **Median Latency:** **{lm['median_ms']} ms**")
    md.append(f"- **P95 Latency:** **{lm['p95_ms']} ms**")

    md.append("\n## 2. Summarization Quality")
    md.append(f"- **Average Summary Length:** {sm['avg_summary_length_chars']} chars")
    md.append(f"- **Average Compression Ratio:** {sm['avg_compression_ratio'] * 100:.1f}% of original body length")
    md.append(f"- **Key Entity/Date Preservation Rate:** **{sm['entity_preservation_rate'] * 100:.1f}%**")

    md.append("\n## 3. Action Extraction & Structured Schema Validity")
    md.append(f"- **Schema Validity Rate:** **{am['structured_schema_validity_rate']}%** (Strict adherence to ALLOWED_ACTION_TYPES)")
    md.append(f"- **Non-Action Rate:** {am['non_action_rate']}%")
    md.append("\n### Action Type Distribution:")
    for act, count in sorted(am["action_type_distribution"].items(), key=lambda x: -x[1]):
        if count > 0:
            md.append(f"- `{act}`: {count} cases")

    md.append("\n## 4. Sample Generations")
    for s in b["sample_generations"][:5]:
        md.append(f"### ID: `{s['id']}` (Intent: `{s['intent']}`, Priority: `{s['priority']}`)")
        md.append(f"- **Subject:** {s['subject']}")
        md.append(f"- **Summary:** {s['summary']}")
        md.append(f"- **Extracted Action:** `{s['action']['action_type']}` -- *{s['action'].get('title') or 'N/A'}* (Due: {s['action'].get('due_date') or 'None'})")

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLAN-T5 generation benchmark")
    parser.add_argument("--dataset-splits", type=str, default="artifacts/canonical_multi_output_dataset.json")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--max-samples", type=int, default=120)
    args = parser.parse_args()

    splits_path = Path(args.dataset_splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits path {splits_path} not found.")

    with splits_path.open("r", encoding="utf-8") as fh:
        split_data = json.load(fh)["splits"]

    # Use validation pool for generation evaluation
    val_ex = [CanonicalEmailExample.from_dict(d) for d in split_data["val"]]
    run_generation_benchmark(val_ex, Path(args.output_dir), max_samples=args.max_samples)


if __name__ == "__main__":
    main()
