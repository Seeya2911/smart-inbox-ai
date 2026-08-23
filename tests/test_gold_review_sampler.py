from ml.gold_review_sampler import build_review_queue


def _row(idx: int, source: str, intent: str, priority: str, score: float) -> dict:
    return {
        "id": f"{source}_{idx}",
        "source": source,
        "subject": f"Subject {idx}",
        "body": f"Body content {idx}",
        "intent": intent,
        "priority": priority,
        "rule_score": score,
        "label_source": "rules",
        "label_confidence": 0.0,
    }


def test_sampler_is_deterministic_and_stratified() -> None:
    rows = []
    idx = 0
    for source in ("enron", "spam", "phishing"):
        for intent, priority, score in (
            ("request", "high", 4.0),
            ("notification", "low", 0.2),
            ("security", "high", 1.8),
        ):
            for _ in range(3):
                rows.append(_row(idx, source, intent, priority, score))
                idx += 1

    first = build_review_queue(rows, count=12, seed=7)
    second = build_review_queue(rows, count=12, seed=7)

    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len(first) == 12
    assert {r["source"] for r in first} == {"enron", "spam", "phishing"}
    assert {r["review_population"] for r in first} == {"high_score", "ambiguous", "low_signal"}
    assert all(r["review_intent"] == "" for r in first)
    assert all(r["review_priority"] == "" for r in first)


def test_sampler_does_not_create_gold_labels() -> None:
    rows = [_row(1, "enron", "request", "high", 4.0)]
    selected = build_review_queue(rows, count=1)
    assert selected[0]["label_source"] == "rules"
    assert selected[0]["label_confidence"] == 0.0
    assert selected[0]["review_intent"] == ""
    assert selected[0]["review_priority"] == ""


def test_sampler_skips_empty_content() -> None:
    rows = [_row(1, "enron", "request", "high", 4.0)]
    rows[0]["body"] = ""
    assert build_review_queue(rows, count=1) == []
