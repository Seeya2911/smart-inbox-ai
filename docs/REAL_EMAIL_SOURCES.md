# Real Email Corpus Expansion

The Smart Inbox training corpus is intentionally multi-source. Enron is useful for corporate/work correspondence, but it is not sufficient coverage for the production intent taxonomy.

## Current real sources

| Source | Role | Production labels assigned here? |
|---|---|---|
| `corbt/enron-emails` | Corporate/work email realism | No |
| `talby/spamassassin` | Spam/ham and unwanted-mail diversity | No |
| `puyang2025/seven-phishing-email-datasets` | Additional independent historical corpora | No |

For the seven-corpus source, the ingestion pipeline keeps only `TREC-05`, `TREC-06`, `TREC-07`, `CEAS-08`, and `Ling`. Its `Enron` and `Assassin` rows are excluded because those source corpora are already ingested separately. This prevents knowingly adding the same source twice.

The source-native binary spam/phishing label is preserved as `source_label` metadata. It is **not** silently converted into Smart Inbox `intent` or `priority`.

## Commands

```bash
python -m ml.email_dataset_ingest --source enron --output data/raw/enron.jsonl --max-rows 10000
python -m ml.email_dataset_ingest --source spam_corpus --output data/raw/spamassassin.jsonl --max-rows 10000
python -m ml.email_dataset_ingest --source phishing_corpus --output data/raw/phishing_corpus.jsonl --max-rows 10000
```

Omit `--max-rows` for the complete available split. Development runs should use a bounded value first.

## Why these sources

The phishing corpus combines seven historical sources and contains more than 200,000 rows in its published train split, including TREC-05/06/07, CEAS-08, Enron, SpamAssassin-derived data, and Ling-Spam. We use it only for the additional independent sources so that the corpus expands in domain and time without intentionally duplicating Enron/SpamAssassin.

## Important boundary

This ingestion stage is **unlabeled**. We do not treat source spam/ham labels as Smart Inbox ground truth. After ingestion we will:

1. deduplicate across sources;
2. inspect quality and source coverage;
3. run the rules only as weak labelers;
4. route ambiguous cases to review;
5. build an isolated human-labeled gold set;
6. train separate intent and priority heads.

Targeted synthetic data remains a later gap-filling step, not the current corpus foundation.
