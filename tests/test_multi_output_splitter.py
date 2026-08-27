from ml.multi_output_splitter import split_multi_output_dataset
from ml.schema import CanonicalEmailExample


def make_examples():
    rows = []
    for i in range(12):
        intent = ["request", "question", "security", "transactional"][i % 4]
        priority = ["medium", "low", "high"][i % 3]
        rows.append(
            CanonicalEmailExample(
                id=f"enron_{i}",
                subject=f"Unique subject {i}",
                body=f"Unique body for example {i} with enough distinct words.",
                intent=intent,
                priority=priority,
                source="enron",
                source_example_id=str(i),
                source_group_id=f"group-{i}",
                label_source="llm",
            )
        )
    return rows


def test_multi_output_split_is_deterministic_and_non_empty():
    examples = make_examples()
    first = split_multi_output_dataset(examples, seed=42)
    second = split_multi_output_dataset(examples, seed=42)

    assert [[e.id for e in split] for split in first] == [[e.id for e in split] for split in second]
    assert all(first)


def test_near_duplicates_stay_in_one_split():
    examples = make_examples()
    examples[1] = CanonicalEmailExample(
        id="enron_1",
        subject="Reset password now",
        body="Please reset your account password immediately before continuing.",
        intent="security",
        priority="high",
        source="enron",
        source_example_id="1",
        source_group_id="security-group",
        label_source="llm",
    )
    examples.append(
        CanonicalEmailExample(
            id="spam_99",
            subject="Reset password now",
            body="Please reset your account password immediately before continuing!",
            intent="security",
            priority="high",
            source="spam_corpus",
            source_example_id="99",
            source_group_id="different-source-group",
            label_source="llm",
        )
    )

    train, val, test = split_multi_output_dataset(examples, seed=7)
    split_of = {}
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        for row in rows:
            split_of[row.id] = name

    assert split_of["enron_1"] == split_of["spam_99"]
