"""Run reproducible multilingual email evaluation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from llm.analyzer import EmailAnalyzer
from llm.provider import MockLLMProvider, OpenAICompatibleProvider

LANGUAGES = ("en", "de", "fr", "es")
LABEL_FIELDS = ("intent", "urgency", "priority")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    """Load the repository's compact JSONL evaluation format."""
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            for required in ("id", "language", "subject", "body", *LABEL_FIELDS):
                if required not in raw:
                    raise ValueError(f"Case {line_number} is missing '{required}'")
            if raw["language"] not in LANGUAGES:
                raise ValueError(f"Unsupported evaluation language: {raw['language']}")
            cases.append({
                "id": raw["id"],
                "language": raw["language"],
                "email": {"subject": raw["subject"], "body": raw["body"]},
                "expected": {field: raw[field] for field in LABEL_FIELDS},
            })
    if not cases:
        raise ValueError("Evaluation corpus is empty")
    return cases


def accuracy(pairs: Iterable[Tuple[str, str]]) -> float:
    pairs = list(pairs)
    return sum(actual == predicted for actual, predicted in pairs) / len(pairs) if pairs else 0.0


def macro_f1(pairs: List[Tuple[str, str]]) -> float:
    labels = sorted({label for pair in pairs for label in pair})
    scores = []
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in pairs)
        fp = sum(actual != label and predicted == label for actual, predicted in pairs)
        fn = sum(actual == label and predicted != label for actual, predicted in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    result: Dict[str, Any] = {}
    for field in LABEL_FIELDS:
        pairs = [(str(r["expected"][field]), str(r["prediction"].get(field, ""))) for r in rows]
        result[field] = {"accuracy": round(accuracy(pairs), 4), "macro_f1": round(macro_f1(pairs), 4)}
    return result


def evaluate(analyzer: EmailAnalyzer, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    errors: List[Dict[str, Any]] = []
    for case in cases:
        try:
            prediction = analyzer.analyze(case["email"]).to_dict()
            grouped[case["language"]].append({"id": case["id"], "expected": case["expected"], "prediction": prediction})
        except Exception as exc:
            errors.append({"id": case["id"], "error": str(exc)})

    report: Dict[str, Any] = {"cases": len(cases), "errors": errors, "languages": {}}
    all_rows = [row for rows in grouped.values() for row in rows]
    for language in LANGUAGES:
        rows = grouped.get(language, [])
        metrics = aggregate(rows)
        language_pairs = [(language, str(r["prediction"].get("language", ""))) for r in rows]
        metrics["language_match"] = round(accuracy(language_pairs), 4)
        metrics["language_disagreements"] = sum(bool(r["prediction"].get("language_disagreement")) for r in rows)
        report["languages"][language] = {"cases": len(rows), "metrics": metrics}
    report["aggregate"] = aggregate(all_rows)
    return report


def build_provider(kind: str):
    return MockLLMProvider() if kind == "mock" else OpenAICompatibleProvider()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="evaluation/multilingual_cases.jsonl")
    parser.add_argument("--provider", choices=("mock", "openai-compatible"), default="mock")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    cases = load_cases(Path(args.corpus))
    report = evaluate(EmailAnalyzer(build_provider(args.provider)), cases)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
