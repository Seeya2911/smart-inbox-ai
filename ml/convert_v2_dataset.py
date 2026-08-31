"""Convert smart_inbox_ai_dataset_v2.xlsx to JSONL for pipeline consumption.

This script performs a deterministic, audit-clean conversion of the v2 XLSX
dataset to newline-delimited JSON so the training pipeline can be run without
Excel as a runtime dependency.

Usage:
    python -m ml.convert_v2_dataset \\
        --input smart_inbox_ai_dataset_v2.xlsx \\
        --output data/smart_inbox_ai_dataset_v2.jsonl

The source XLSX file is NEVER modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert smart_inbox_ai_dataset_v2.xlsx → JSONL"
    )
    parser.add_argument(
        "--input", type=str,
        default="smart_inbox_ai_dataset_v2.xlsx",
        help="Path to the source XLSX file (never modified)"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/smart_inbox_ai_dataset_v2.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--no-audit", action="store_true",
        help="Skip printing the audit report (faster)"
    )
    args = parser.parse_args()

    from ml.v2_dataset_loader import export_to_jsonl, load_v2_dataset

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading: {input_path}")
    examples, audit = load_v2_dataset(input_path, print_audit=not args.no_audit)
    print(f"Valid examples: {len(examples)}")
    print(f"Rejected rows : {audit['rejected_rows']}")

    export_to_jsonl(examples, output_path)
    print(f"Exported {len(examples)} examples -> {output_path}")

    # Write audit report alongside the JSONL
    audit_path = output_path.with_suffix(".audit.json")
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Audit report  -> {audit_path}")


if __name__ == "__main__":
    main()
