# Smart Inbox AI: Multi-Output Priority & Intent NLP Pipeline

Smart Inbox AI provides a reproducible, supervised machine learning architecture predicting both **INTENT** and **PRIORITY** independently.

## Multi-Output Architecture Principles

1. **Independent Dual Predictions**: `INTENT` and `PRIORITY` are predicted by two separate classification heads rather than a single merged tuple tag.
2. **Rules as Weak Labelers**: Output from `priority_tagging.py` is treated strictly as an uncalibrated heuristic `rule_score`. High rule-score examples are routed as pseudo-labeled candidates, ambiguous examples flow to an LLM/human review queue, and low-signal records are retained in an unlabeled pool.
3. **Dual Weak Supervision**: `ml.weak_labeler` combines `priority_tagging.py` (priority rules) and `ml.intent_rules` (intent rules) to generate pseudo-labels for both targets simultaneously.
4. **Boilerplate & Quoted-Chain Stripping**: Signature blocks, quoted reply chains (`> ...`), forward headers, and legal disclaimers are stripped BEFORE hashing or deduplication.
5. **ID Namespacing**: Every example ID is explicitly prefixed by its source (`enron_*`, `spam_*`, `synthetic_*`, `inbox_*`) to guarantee global uniqueness across merged corpora.
6. **Gold Set Isolation Invariant**: `gold/test.jsonl` is a hand-labeled, untouched test set that is **NEVER** used during model training, hyperparameter tuning, or pseudo-labeling. Zero ID or text overlap is strictly enforced by unit tests.

---

## Canonical Taxonomy

### Intent Categories
- `SECURITY` (Password resets, 2FA alerts, suspicious login notifications)
- `TRANSACTIONAL` (Order confirmations, receipts, invoices, shipping tracking)
- `MEETING` (Calendar invites, reschedule requests, sync proposals)
- `REQUEST` (Action requests, document reviews, task assignments)
- `QUESTION` (Inquiries, clarification requests, help queries)
- `NOTIFICATION` (System updates, maintenance notices, automated digests)
- `PROMOTION` (Discounts, sales, marketing newsletters)
- `COMPLAINT` (Service issues, dissatisfaction, refund demands)
- `FOLLOW_UP` (Thread follow-ups, reminders, status checks)
- `INFORMATION` (FYI messages, reports, general announcements)
- `OTHER` (General fallback)

### Priority Categories
- `HIGH` (Urgent action / security outage / time-sensitive deadline)
- `MEDIUM` (Standard meeting / input request)
- `LOW` (Transactional receipt / marketing promotion / system digest)

---

## Reproducible CLIs

### 1. Noisy Synthetic Gap Generator (`ml.generate_synthetic`)

Generate targeted synthetic examples for categories under-represented in corporate mail:

```bash
python -m ml.generate_synthetic --output artifacts/synthetic_gaps.jsonl --count 100 --seed 42
```

### 2. Dual Weak Labeler & Router CLI (`ml.weak_labeler`)

Route raw or synthetic emails into High Rule Score, Ambiguous, and Low Signal pools:

```bash
python -m ml.weak_labeler --input artifacts/synthetic_gaps.jsonl --output artifacts/weak_labeled.jsonl
```

### 3. Gold Set Splitter (`ml.gold_labeler`)

Create isolated gold splits (`gold/train.jsonl`, `gold/val.jsonl`, `gold/test.jsonl`):

```bash
python -m ml.gold_labeler --input artifacts/weak_labeled.jsonl --output-dir gold --seed 42
```

### 4. Multi-Output Baseline Trainer (`ml.train_multi_output`)

Train the mandatory **TF-IDF + Logistic Regression** baseline model:

```bash
python -m ml.train_multi_output \
  --data artifacts/weak_labeled.jsonl \
  --output-model artifacts/multi_output_model.joblib \
  --seed 42
```

### 5. Multi-Output Evaluation Harness (`ml.evaluate_multi_output`)

Evaluate strictly on untouched gold test set (`gold/test.jsonl`) with per-source performance breakdowns:

```bash
python -m ml.evaluate_multi_output \
  --data gold/test.jsonl \
  --model artifacts/multi_output_model.joblib \
  --output artifacts/multi_output_eval_results.json
```
