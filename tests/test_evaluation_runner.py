from pathlib import Path

from evaluation.runner import evaluate, load_cases
from llm.analyzer import EmailAnalyzer
from llm.provider import MockLLMProvider


CORPUS = Path("evaluation/multilingual_cases.jsonl")


def test_corpus_loads_with_four_languages():
    cases = load_cases(CORPUS)
    assert len(cases) == 12
    assert {case["language"] for case in cases} == {"en", "de", "fr", "es"}


def test_evaluation_report_has_per_language_metrics():
    cases = load_cases(CORPUS)
    report = evaluate(EmailAnalyzer(MockLLMProvider()), cases)
    assert set(report["languages"]) == {"en", "de", "fr", "es"}
    assert report["cases"] == 12
    assert report["errors"] == []
    for language in report["languages"].values():
        assert language["cases"] == 3
        assert 0.0 <= language["metrics"]["intent"]["accuracy"] <= 1.0
        assert 0.0 <= language["metrics"]["urgency"]["macro_f1"] <= 1.0
