# Real LLM evaluation

This directory contains the reproducible evaluation corpus and runner for Smart Inbox AI.

## Important distinction

`MockLLMProvider` is deterministic test infrastructure. It is **not an LLM** and its numbers must never be presented as LLM performance.

A real run uses `OpenAICompatibleProvider` and the same 12-case synthetic multilingual corpus used by the baseline comparison.

## Run with a real provider

1. Create a local virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. Never commit `.env` or real mailbox data.
3. Run:

```bash
python -m evaluation.runner --provider openai-compatible --corpus evaluation/multilingual_cases.jsonl --output evaluation/results-real.json
```

The command sends **only the version-controlled synthetic evaluation corpus** to the configured provider. Do not point this command at a real mailbox export without an explicit privacy/data-processing review.

## What the report contains

- Intent, urgency, and priority accuracy and macro-F1.
- Results for English, German, French, and Spanish separately.
- Aggregate results across all cases.
- Language-detector/LLM agreement signals.
- Provider errors, if any.
- The same-corpus legacy keyword baseline.
- Non-secret run metadata such as provider type, model, endpoint, and whether an API key was configured.

The runner does **not** write the API key into the report.

## Interpreting results

The current corpus contains only 12 synthetic cases (3 per language). This is a development/evaluation fixture, not evidence of generalisation to real-world email.

Do not claim that the LLM is superior based on this corpus alone. Use the results to identify failure modes and expand the labelled evaluation set before making research claims.

For academic reporting, record at minimum:

- provider and exact model identifier
- evaluation corpus version/commit
- prompt version
- temperature/configuration
- per-language metrics
- aggregate metrics
- failures and disagreements
- whether inference was hosted or local
