# Smart Inbox AI Dataset Protocol

## Purpose

This document defines how training, development, and final evaluation data are kept separate. The goal is to make reported model performance reproducible and to prevent source or translation leakage.

## Task taxonomy

The project evaluates three application outputs:

- **Intent**: the primary communicative purpose of the email.
- **Urgency**: how quickly the recipient needs to act, based on context rather than isolated words.
- **Priority**: the operational importance of the message after considering urgency, consequences, requested action, and context.

The current intent taxonomy is intentionally small and application-oriented:

`REQUEST`, `QUESTION`, `INFORMATION`, `COMPLAINT`, `MEETING`, `INVITATION`, `FOLLOW_UP`, `TRANSACTION`, `NOTIFICATION`, `OTHER`.

Urgency and priority use the ordered labels `LOW`, `NORMAL`, `HIGH`, and `CRITICAL`.

These definitions are project labels. They must not be retroactively imposed on public datasets when the source dataset does not contain an equivalent annotation.

## Data tiers

### Tier A — email-specific supervised data

The manually verified Enron Intent Dataset is the primary public source for learning an **action/response intent** signal. It contains 3,655 verified email sentences with positive/negative intent labels. The source README states that the labels cover genuine actions, future meetings/events, and questions judged to require a response, while also documenting subjective judgement and assumptions. Because this is a binary intent signal, it is mapped only to the project's `REQUEST`/`INFORMATION` family where the mapping is explicitly justified; it is **not** used to manufacture urgency or priority labels.

Source: https://github.com/Charlie9/enron_intent_dataset_verified

### Tier B — multilingual auxiliary NLU data

AmazonScience/MASSIVE is used only as an auxiliary multilingual intent source. MASSIVE contains more than one million utterances across 52 languages and 60 intents, but it was created from localized voice-assistant interactions, not email. Only its explicitly email-related intents are eligible for the adapter. Source labels and provenance fields must remain attached to every exported row.

Source: https://huggingface.co/datasets/AmazonScience/massive

MASSIVE examples must never be described as real email examples in reported results.

### Tier C — controlled project data

Urgency and priority require labels that the public sources above do not provide in a defensible, task-aligned form. Controlled project examples may therefore be created and manually reviewed for these labels. They must be explicitly marked `synthetic` and must not be presented as naturally occurring email data.

### Tier D — final gold evaluation set

The final evaluation set is held out from all training and model-selection activity. It should contain independently authored examples in English, German, French, and Spanish, including implicit urgency, negation, deadlines, politeness, indirect requests, and ambiguous cases. Synthetic evaluation examples are acceptable as a controlled robustness suite, but they are not evidence of broad real-world generalisation.

## Leakage controls

1. Never train on `evaluation/multilingual_cases.jsonl` or the future gold evaluation set.
2. Keep source dataset train/dev/test partitions intact when a source provides them.
3. Do not place translated or parallel versions of the same source example across train and test.
4. Deduplicate normalized subject/body text before splitting.
5. Preserve source IDs so duplicate and near-duplicate investigations are possible.
6. If augmentation creates a family of related examples, assign the whole family to one split.
7. Do not use test-set performance to choose labels, prompts, thresholds, or hyperparameters.
8. Keep public source data out of Git unless its redistribution terms clearly permit it and the repository explicitly records the provenance/license.

## Reproducibility metadata

Every generated training/evaluation artifact should record:

- source dataset and version/revision
- source split
- source example ID
- project label mapping, if any
- language/locale
- whether the example is `public`, `synthetic`, or `gold`
- preprocessing version
- random seed
- train/dev/test split strategy
- model identifier and revision

## Reporting rules

Reported accuracy and macro-F1 must identify the exact test set. Results from the 12-case multilingual smoke corpus are engineering checks, not benchmark claims. If a source dataset is out-of-domain, results on that source must be described as auxiliary/domain-transfer evidence rather than email performance.

The first meaningful Smart Inbox comparison should report the same held-out test set for:

1. deterministic keyword baseline;
2. TF-IDF + linear classifier baseline;
3. fine-tuned multilingual transformer;
4. optional hosted/local LLM, when available.

No result may be invented to fill a table.