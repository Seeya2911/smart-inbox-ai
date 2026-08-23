from ml.data_quality import check_split_leakage
from ml.dataset_splitter import _near_duplicate_group_ids, split_intent_dataset
from ml.schema import CanonicalIntentExample


def make_example(idx: int, text: str, intent: str) -> CanonicalIntentExample:
    return CanonicalIntentExample(
        text=text,
        language="en",
        canonical_intent=intent,
        source_dataset="test",
        source_example_id=str(idx),
        original_label=intent,
        source_group_id=f"group-{idx}",
    )


def test_near_duplicate_variants_share_a_cluster():
    examples = [
        make_example(0, "do i have any new email", "information"),
        make_example(1, "do i have any new emails", "information"),
        make_example(2, "olly do i have any emails from robert", "information"),
        make_example(3, "do i have any emails from robert", "information"),
        make_example(4, "alexa add jpearsonjessica at gmail dot com to email contacts", "request"),
        make_example(5, "add jpearsonjessica at gmail dot com to email contacts", "request"),
        make_example(6, "send the report to sam", "request"),
    ]

    groups = _near_duplicate_group_ids(examples, threshold=0.85)

    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[4] == groups[5]
    assert groups[0] != groups[6]


def test_split_keeps_near_duplicates_in_one_split():
    texts = [
        ("do i have any new email", "information"),
        ("do i have any new emails", "information"),
        ("are there any unread messages", "information"),
        ("are there any unread messages now", "information"),
        ("olly do i have any emails from robert", "information"),
        ("do i have any emails from robert", "information"),
        ("send the report to sam", "request"),
        ("send the report to alex", "request"),
        ("please send the report to sam", "request"),
        ("please send the report to alex", "request"),
        ("email the invoice to sam", "request"),
        ("email the invoice to alex", "request"),
    ]
    examples = [make_example(idx, text, intent) for idx, (text, intent) in enumerate(texts)]

    train, val, test = split_intent_dataset(examples, seed=42)
    assignments = {
        ex.source_example_id: split
        for split, items in {"train": train, "val": val, "test": test}.items()
        for ex in items
    }

    assert assignments["0"] == assignments["1"]
    assert assignments["2"] == assignments["3"]
    assert assignments["4"] == assignments["5"]
    assert assignments["6"] == assignments["8"]
    assert assignments["7"] == assignments["9"]
    check_split_leakage(train, val, test)
