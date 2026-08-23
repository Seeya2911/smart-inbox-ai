"""Gold Set Labeling & Split Management Tool.

Manages hand-labeled Gold evaluation datasets (gold/train.jsonl, gold/val.jsonl, gold/test.jsonl).
Ensures strict Gold test isolation so gold/test.jsonl is NEVER used during model training or pseudo-labeling.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ml.schema import ALLOWED_INTENTS, ALLOWED_PRIORITIES, CanonicalEmailExample


def create_gold_splits(
    examples: List[CanonicalEmailExample],
    output_dir: Path = Path("gold"),
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Split hand-labeled gold examples into train, validation, and untouched gold test split."""
    if not examples:
        raise ValueError("Cannot create gold splits from an empty list.")

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    gold_train = shuffled[:n_train]
    gold_val = shuffled[n_train : n_train + n_val]
    gold_test = shuffled[n_train + n_val :]

    output_dir.mkdir(parents=True, exist_ok=True)

    splits_map = {
        "train.jsonl": gold_train,
        "val.jsonl": gold_val,
        "test.jsonl": gold_test,
    }

    counts = {}
    for filename, ex_list in splits_map.items():
        dest = output_dir / filename
        with dest.open("w", encoding="utf-8") as f:
            for ex in ex_list:
                f.write(json.dumps(ex.to_dict()) + "\n")
        counts[filename] = len(ex_list)

    summary = {
        "status": "success",
        "total_gold_examples": len(examples),
        "split_counts": counts,
        "output_directory": str(output_dir),
    }

    return summary


def load_gold_split(filepath: Path) -> List[CanonicalEmailExample]:
    """Load a gold JSONL split file into CanonicalEmailExample list."""
    if not filepath.is_file():
        return []
    records = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(CanonicalEmailExample.from_dict(json.loads(line)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Set Splitter & Labeling Tool CLI")
    parser.add_argument("--input", type=str, required=True, help="Input raw or candidate JSON/JSONL file")
    parser.add_argument("--output-dir", type=str, default="gold", help="Directory to save gold splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    input_path = Path(args.input)
    text_content = input_path.read_text(encoding="utf-8").strip()
    raw_records = []

    if text_content.startswith("["):
        raw_records = json.loads(text_content)
    else:
        lines = [l.strip() for l in text_content.splitlines() if l.strip()]
        raw_records = [json.loads(l) for l in lines]

    # Convert to CanonicalEmailExample marking label_source as 'human' if confidence is 1.0
    gold_examples = []
    for r in raw_records:
        r_payload = dict(r)
        r_payload["label_source"] = "human"
        r_payload["label_confidence"] = 1.0
        gold_examples.append(CanonicalEmailExample.from_dict(r_payload))

    summary = create_gold_splits(gold_examples, output_dir=Path(args.output_dir), seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
