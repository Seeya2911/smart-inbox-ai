# Smart Inbox AI — Multi-Output Data and Labeling Guidelines

## Status

Version: `v1`

This document defines the training target for the Smart Inbox ML system. It deliberately separates **intent** from **priority**. The two labels answer different questions and must not be collapsed into a single class.

The existing two-class `information/request` auxiliary benchmark remains useful for validating the ML pipeline, but it is **not** the target production taxonomy.

## Core rules

1. Do not use synthetic text as a replacement for real email data. Synthetic examples are gap-fillers for categories that public corpora do not cover well.
2. Rule-based outputs are weak labels. A rule score is a heuristic signal, not a probability or ground truth.
3. Human-reviewed examples are the strongest supervised labels in the initial dataset.
4. LLM-reviewed labels are reviewed labels, not automatically trusted ground truth. They must be tracked separately from human labels.
5. User feedback is behavioral evidence and candidate labeling data; it is not automatically a correct label.
6. Every training example must preserve provenance and label provenance.
7. The protected gold test set must never be pseudo-labeled, used for training, or used to tune thresholds.
8. Duplicate and near-duplicate content must remain isolated across splits.

## Target labels

### Intent

Every email receives exactly one **primary intent** for the first supervised model.

| Label | Definition | Positive examples | Do not use for |
|---|---|---|---|
| `request` | The sender is asking the recipient to perform an action. | "Please send me the signed form." | Pure questions that only seek information. |
| `question` | The primary purpose is to obtain information or an answer. | "What time does the meeting start?" | Requests that clearly ask the recipient to perform an action. |
| `meeting` | Scheduling, changing, cancelling, or coordinating a meeting/appointment/call is the primary purpose. | "Can we move our meeting to 3pm?" | General requests that mention a meeting incidentally. |
| `notification` | The email primarily informs the recipient that something happened or changed, without requiring a substantive response. | "Your package has shipped." | Security incidents requiring action. |
| `promotion` | Marketing, advertising, sales, offers, newsletters, or promotional content is the primary purpose. | "20% off this weekend." | Transactional receipts/order confirmations that are not marketing. |
| `complaint` | The sender is expressing dissatisfaction or reporting a service/product problem as the primary purpose. | "The replacement you sent is still broken." | Neutral support requests without a complaint component. |
| `follow_up` | The email primarily follows up on an earlier request, commitment, conversation, or unresolved issue. | "Following up on my application from last week." | A new request that merely references an earlier conversation. |
| `information` | The primary purpose is to provide or share useful information rather than request action. | "Here are the documents from yesterday's call." | Notifications about a discrete event when notification is clearly primary. |
| `other` | The email cannot be assigned to one of the defined intents without forcing a weak interpretation. | Ambiguous or genuinely out-of-taxonomy content. | Do not use merely because labeling is inconvenient. |

### Intent tie-breaking rules

Emails can contain multiple intents. The model currently predicts one primary intent, so annotators must use a consistent hierarchy based on the **main communicative purpose**:

1. `meeting` when scheduling/coordinating a meeting is the central action.
2. `complaint` when dissatisfaction/problem reporting is central.
3. `follow_up` when the central purpose is pursuing an unresolved prior interaction.
4. `request` when the sender primarily wants the recipient to do something.
5. `question` when the sender primarily wants an answer.
6. `promotion` when commercial/promotional messaging is central.
7. `notification` when the sender primarily reports an event/state change.
8. `information` when the sender primarily shares information.
9. `other` only when none of the above is defensible.

This ordering is a tie-breaker, not a claim that one intent is inherently more important than another.

### Priority

Priority is independent of intent and reflects **how important and time-sensitive it is for the recipient to act or pay attention**, given the email context.

| Label | Definition | Typical evidence |
|---|---|---|
| `high` | Failure to act or pay attention soon could cause meaningful harm, loss, security impact, missed critical deadline, or serious consequence. | Active security compromise, deadline today, critical operational interruption, immediate action required. |
| `medium` | Worth timely attention, but delay is unlikely to cause serious harm. | Routine work request, upcoming meeting change, non-critical account issue, ordinary follow-up. |
| `low` | No meaningful time pressure or consequence from delaying attention. | Routine notification, ordinary receipt, newsletter, promotion, general information. |

Priority must be assigned from context rather than keywords. The presence of words such as `urgent`, `important`, or `ASAP` is evidence, not a label by itself.

Examples:

- "Your order has shipped." → intent `notification`, priority `low`.
- "Your password was changed. If this wasn't you, secure your account now." → intent `notification`, priority `high`.
- "Can we move tomorrow's meeting to 3pm?" → intent `meeting`, priority `medium` unless the email establishes a higher/lower consequence.
- "URGENT: 50% off today only!" → intent `promotion`, priority `low` unless the recipient's context establishes a real consequence.

## Required dataset record

The canonical record should preserve enough information to reconstruct how a label was obtained:

```json
{
  "id": "stable-example-id",
  "subject": "Email subject",
  "body": "Email body",
  "intent": "request",
  "priority": "medium",
  "priority_reasons": ["action_required"],
  "source": "enron",
  "source_example_id": "original-id",
  "source_split": "train",
  "label_source": "human",
  "label_confidence": 1.0,
  "rule_score": null,
  "is_synthetic": false,
  "language": "en",
  "provenance": "human-reviewed-v1"
}
```

### Label provenance

Allowed `label_source` values should distinguish at least:

- `human` — directly reviewed by a human annotator.
- `llm` — assigned by an LLM review process and not yet human-confirmed.
- `rules` — generated by the deterministic weak-label system.
- `user_feedback` — supplied as an explicit user correction.
- `mixed` — a later adjudicated label based on multiple sources.

`label_confidence` is only meaningful when its definition is documented for the relevant label source. The current rule tagger's numeric score must remain in `rule_score`; it must not be renamed to confidence without calibration evidence.

## Data-source policy

### Real public data

Prefer real email corpora for general language variation and realistic formatting. Track the original dataset and original identifier for every row.

### Synthetic data

Synthetic examples are allowed only to fill documented coverage gaps such as modern security alerts, 2FA messages, receipts, invoices, and other patterns absent from older public corpora. Synthetic data must be explicitly marked `is_synthetic: true` and `source: synthetic`.

Synthetic generation should deliberately include realistic variation: short and long messages, typos, signatures, forwarded/replied content, vague language, mixed intents, contradictory urgency signals, false urgency words, and urgent cases with no obvious urgency keyword.

### User inbox

Real mailbox data must not be committed to the repository. It should be processed locally through the future IMAP ingestion path and must retain privacy controls and consent requirements.

## Gold evaluation set

Create the gold set before bulk pseudo-labeling.

Recommended eventual size: approximately 300–500 carefully reviewed emails, with coverage across every intent and priority class. The gold test split must be immutable for model development.

Required discipline:

- `gold/train` may be used for supervised training/checks.
- `gold/validation` may be used for model/threshold selection.
- `gold/test` is evaluation-only.
- Rules, LLM pseudo-labeling, threshold tuning, and model training must not inspect `gold/test` labels.

Evaluation must report overall and per-class metrics, including macro F1, and should be broken down by data source when enough examples exist.

## Weak-label workflow

1. Run deterministic rules and store the resulting `rule_score` and reasoning.
2. High-score examples become pseudo-label candidates only after the score has been validated against a human-reviewed sample.
3. Ambiguous examples go to an LLM/human review queue.
4. Low-signal examples remain in an unlabeled pool; do not silently discard them.
5. Ensure a meaningful portion of reviewed ambiguous examples enters training so the model does not simply learn the obvious rule patterns.
6. Preserve all label-source metadata so experiments can compare rule, LLM, human, and user-feedback supervision.

## What is explicitly deferred

Reinforcement learning, Q-learning, online learning, and preference optimization are not part of the initial training loop. They require enough real interaction/reward data to justify their complexity. Existing feedback/Q-learning components may remain in the repository but must not contaminate the initial supervised benchmark.
