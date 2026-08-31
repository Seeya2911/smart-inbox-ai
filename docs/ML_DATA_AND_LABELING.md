# Smart Inbox AI — Multi-Output Data and Labeling Guidelines

## Status

Version: `v2`

This document defines the training target for the Smart Inbox ML system. It deliberately separates **intent** from **priority**. The two labels answer different questions and must not be collapsed into a single class.

The historical two-class `information/request` auxiliary benchmark remains useful for pipeline regression tests, but it is **not** the target production taxonomy.

## Core rules

1. Real public email corpora are the primary data source. Synthetic text is only a documented gap-filler.
2. Intent and priority are two independent predictions.
3. The LLM is the **authoritative teacher** for the initial pseudo-labeled corpus.
4. Rule-based predictions are **not ground truth and never override the LLM**. They are retained as an independent comparison signal and disagreement feature.
5. When rules and the LLM disagree, the LLM label is used for training and `label_resolution_reason` records why the LLM interpretation won.
6. The LLM's self-reported confidence is a teacher certainty signal, not a calibrated probability. It must not be treated as measured model accuracy.
7. No human-review stage is required for the initial corpus-generation pipeline. Human labels can be introduced later as an independent validation set if needed.
8. User feedback is behavioral evidence/candidate labeling data; it is not automatically a correct label.
9. Every training example must preserve source and label provenance.
10. Duplicate and near-duplicate content must remain isolated across splits.
11. The LLM-labeled test split is a **teacher-agreement benchmark**, not a ground-truth benchmark. Do not report it as human accuracy.

## Target labels

### Intent

Every email receives exactly one **primary intent** for the first supervised model.

Allowed labels:

- `request` — sender asks recipient to perform an action.
- `question` — primary purpose is obtaining information or an answer.
- `meeting` — scheduling/changing/cancelling/coordinating a meeting or appointment.
- `notification` — automated/system update, digest, newsletter, or policy notice.
- `promotion` — marketing, advertising, sales, offers, newsletters, or promotional content.
- `complaint` — dissatisfaction or service/product problem is central.
- `follow_up` — follows up on an earlier request, commitment, conversation, or unresolved issue.
- `information` — primarily shares useful information without a clear request/question.
- `security` — authentication, account security, suspicious activity, password/2FA/security events.
- `transactional` — purchases, receipts, invoices, payments, shipping, orders.
- `other` — none of the above is defensible.

### Intent tie-breaking

When an email contains multiple intents, choose the primary communicative purpose in this order:

1. `meeting`
2. `complaint`
3. `follow_up`
4. `request`
5. `question`
6. `promotion`
7. `notification`
8. `information`
9. `other`

This is a tie-breaker, not an importance ranking.

### Priority

Priority is independent of intent and reflects how important and time-sensitive it is for the recipient to act or pay attention.

- `high` — delay could cause meaningful harm, loss, security impact, critical deadline miss, or serious consequence.
- `medium` — worth timely attention, but delay is unlikely to cause serious harm.
- `low` — no meaningful time pressure or consequence from delaying attention.

Priority must be contextual. `urgent`, `important`, and `ASAP` are evidence, not labels by themselves.

Examples:

- "Your order has shipped." → `notification`, `low`.
- "Your password was changed. If this wasn't you, secure your account now." → `security` or `notification` depending on the primary communicative purpose, `high`.
- "Can we move tomorrow's meeting to 3pm?" → `meeting`, usually `medium`.
- "URGENT: 50% off today only!" → `promotion`, usually `low`.

## Canonical dataset record

```json
{
  "id": "enron_123",
  "subject": "Email subject",
  "body": "Email body",
  "intent": "request",
  "priority": "medium",
  "priority_reasons": ["action_required"],
  "source": "enron",
  "source_example_id": "123",
  "source_split": "train",
  "label_source": "llm",
  "label_confidence": 0.92,
  "rule_score": 7.0,
  "rule_intent": "request",
  "rule_priority": null,
  "llm_rule_agreement": true,
  "llm_intent_reason": "The sender asks the recipient to act.",
  "llm_priority_reason": "The requested action is useful but not time-critical.",
  "label_resolution_reason": "LLM and rule signals agree; the LLM remains authoritative.",
  "is_synthetic": false,
  "language": "en",
  "provenance": "corbt/enron-emails"
}
```

`source_example_id` and `source_split` must survive every conversion stage so a labeled row can be traced back to its raw source.

## Labeling workflow

1. Ingest real corpora without assigning Smart Inbox labels.
2. Deduplicate and perform coverage analysis before labeling.
3. Export the clean corpus to the external labeling format (`subject`, `body`, `source`) while retaining a local manifest that maps workbook rows back to source identity.
4. Run the LLM teacher over the complete corpus. The LLM emits exactly one intent, one priority, reasons, and a self-reported certainty score.
5. Run the deterministic rule engine independently on the same email.
6. Keep the LLM label whenever the two disagree. Preserve the rule output and an explicit resolution reason; never silently overwrite the disagreement.
7. Keep all LLM-labeled rows, including low-confidence or rule-disagreement cases. Do not train only on easy rule matches.
8. Split the resulting canonical corpus using group-aware and near-duplicate-aware logic before model training.
9. Train the lightweight student model on the training split only.
10. Use validation for model/threshold selection. Keep the test split untouched by training.

## What the evaluation means

Because the initial labels are generated by the same LLM teacher used to create the training corpus, the resulting test set does **not** provide independent truth. Student metrics on that split answer:

> "How well does the lightweight classifier reproduce the teacher's labeling policy on unseen, leakage-isolated emails?"

They do **not** answer:

> "How accurate is the classifier on objectively correct human labels?"

For a defensible real-world accuracy claim, add an independent human-reviewed test set later or use an independently sourced labeling process. This is not part of the initial automated labeling loop.

## Data-source policy

Prefer real email corpora for language variation and realistic formatting. Track original dataset and original identifier for every row.

Synthetic examples are allowed only to fill documented gaps such as modern security alerts, 2FA messages, receipts, invoices, and other patterns absent from older public corpora. Mark them explicitly with `is_synthetic: true` and `source: synthetic`.

Synthetic generation should include realistic variation: short/long messages, typos, signatures, forwarded/replied content, vague language, mixed intents, contradictory urgency signals, false urgency words, and urgent cases with no obvious urgency keyword.

User inbox data must not be committed to the repository. It should be processed locally through the future IMAP ingestion path with privacy controls and consent.

## Deferred learning loop

Reinforcement learning, Q-learning, online learning, and preference optimization are **not** part of the initial training loop. They require enough real interaction/reward data to justify their complexity.

The future loop is:

`IMAP -> ingestion -> LLM/rule teacher pipeline -> student classifier -> prediction -> explicit user correction/behavioral evidence -> curated feedback dataset -> retraining`

Existing feedback/Q-learning components may remain in the repository but must not contaminate the initial supervised benchmark.
