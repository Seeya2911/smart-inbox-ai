# Real LLM evaluation

This directory contains the reproducible evaluation corpus and runner for Smart Inbox AI.

## Important distinction

`MockLLMProvider` is deterministic test infrastructure. It is **not an LLM** and its numbers must never be presented as LLM performance.

A real run uses `OpenAICompatibleProvider` and the same 12-case synthetic multilingual corpus used by the baseline comparison.

## Easiest OpenAI setup

For a direct OpenAI run, **only `OPENAI_API_KEY` is required**. The provider uses the standard OpenAI API endpoint and `gpt-5.6` by default. You can override the model or endpoint with `OPENAI_MODEL` and `OPENAI_BASE_URL` if needed.

### Windows PowerShell

From the repository root:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python -m evaluation.runner --provider openai-compatible --corpus evaluation/multilingual_cases.jsonl --output evaluation/results-real.json
```

The key exists only in the current PowerShell session. It is not written to the repository.

### Optional: persistent user-level environment variable

If you prefer not to set the variable every time, Windows can store it for your user account:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your_api_key_here", "User")
```

Open a **new** terminal after doing this. Do not put the key in a committed `.env` file or source code.

### Optional `.env` workflow

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and load it with your preferred environment-variable loader. The repository does not require `.env` loading for the evaluation command itself; environment variables are the supported runtime interface.

## Choosing a model

The default is `gpt-5.6`. To use another model available to your API project:

```powershell
$env:OPENAI_MODEL="your_model_id"
```

The API can list models available to your project; availability and permissions can vary by account.

## What the evaluation does

The command sends **only the version-controlled synthetic evaluation corpus** to the configured provider. It does not read your mailbox. Do not point this command at a real mailbox export without an explicit privacy/data-processing review.

The report contains:

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
