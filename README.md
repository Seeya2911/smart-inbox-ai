# Smart Inbox AI

A research-oriented email intelligence prototype that combines **large language model (LLM) analysis**, deterministic NLP baselines, user feedback, prioritisation, and an interactive Streamlit interface.

> **Academic scope:** This repository is a research and engineering prototype, not a production email security, autonomous-agent, or enterprise mail-management system.

## Why this project is AI/LLM-based

The current architecture separates semantic analysis from deterministic baseline logic. The LLM layer produces validated structured analysis of an email, while the historical rule-based summarizer remains available as a reproducible baseline and fallback.

```text
                         Email
                           │
                    normalization
                           │
              ┌────────────┴────────────┐
              │                         │
      Deterministic baseline       LLM analysis
      SmartBrief v3                LLMProvider
              │                  ┌──────┴──────┐
              │                  │             │
              │              Hosted LLM    Local LLM
              │              / compatible   / Ollama
              │                  │             │
              └────────────┬─────┴─────────────┘
                           │
                 Structured EmailAnalysis
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          summary       intent       urgency
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                priority / suggestions
                           │
                     user feedback
                           │
                       evaluation
```

The project therefore supports a meaningful experimental comparison rather than treating keyword matching as an LLM.

## Research questions

The repository is designed around questions such as:

1. Does LLM-based semantic analysis improve email intent and urgency classification compared with a deterministic baseline?
2. How does structured-output validation affect robustness of downstream email workflows?
3. What are the trade-offs between hosted and local LLM inference in latency, privacy, and reproducibility?
4. Can user feedback improve prioritisation without silently changing the semantic analysis model?

These questions are intentionally more modest than claiming general-purpose autonomous email intelligence.

## Architecture

### LLM layer

The `llm/` package contains:

- `provider.py` — provider abstraction and OpenAI-compatible client support.
- `schemas.py` — validated structured `EmailAnalysis` output.
- `analyzer.py` — orchestration of prompt construction and provider output.
- `__init__.py` — public package interface.

The provider boundary allows hosted APIs, OpenAI-compatible local endpoints, and deterministic mock inference to share the same application contract.

### Deterministic baseline

`smart_summarizer_v3.py` is explicitly retained as a **baseline**, not presented as an LLM. It provides reproducible summarisation, intent classification, urgency analysis, and context handling for regression tests and future comparative evaluation.

### Feedback and personalisation

The existing feedback and priority components are kept separate from semantic LLM inference. This prevents user-specific preferences from being confused with model capability when evaluating results.

## Privacy and responsible AI

Email content can contain highly sensitive personal, academic, financial, or professional information. The repository therefore follows a privacy-first development model:

- API keys belong in environment variables, never source code.
- `.env` files and local credentials are ignored by Git.
- A deterministic mock provider supports credential-free development and CI.
- A local OpenAI-compatible endpoint can be configured when email content must remain on-device.
- Real mailbox data should not be committed to the repository.
- The application should obtain appropriate user consent before processing real mailbox content.

No claim is made that an external LLM provider is automatically GDPR-compliant for every deployment; deployment-specific data-processing and retention requirements must be assessed separately.

## Reproducible development

### 1. Environment

```bash
git clone https://github.com/Seeya2911/smart-inbox-ai.git
cd smart-inbox-ai
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Credential-free demo

The mock provider is the default path for development and tests. This makes the core analysis architecture runnable without a paid API account or real mailbox.

```bash
python main.py
```

### 3. Configure an LLM

Copy `.env.example` to `.env` and provide the required provider configuration. Do **not** commit `.env`.

The implementation is intentionally compatible with OpenAI-style endpoints so a hosted provider or a local model server can be selected through configuration.

### 4. Dashboard

```bash
streamlit run dashboard.py
```

A lightweight demo interface is also available:

```bash
streamlit run demo_streamlit_app.py
```

## Testing and CI

Run the complete test suite locally:

```bash
python -m compileall -q .
python -m pytest -q
```

GitHub Actions also compiles the repository and executes the tests on pushes and pull requests.

The test suite includes both the modern LLM analysis layer and regression coverage for the deterministic baseline.

## Evaluation plan

The project should report LLM results separately from historical rule-based results. A future evaluation run should use a fixed, documented test set and report at least:

| Dimension | Example measure |
|---|---|
| Intent classification | Accuracy / macro-F1 |
| Urgency classification | Accuracy / macro-F1 |
| Structured output | Schema-valid response rate |
| Summary quality | Human rating or documented rubric |
| Robustness | Invalid-output / retry rate |
| Efficiency | Median and p95 latency |
| Privacy | Hosted vs local data-flow description |
| Reproducibility | Provider/model/version/configuration |

No benchmark number should be claimed until it has been measured on a documented evaluation set.

## Repository structure

```text
smart-inbox-ai/
├── llm/
│   ├── analyzer.py
│   ├── provider.py
│   └── schemas.py
├── tests/
├── main.py
├── app.py
├── dashboard.py
├── demo_streamlit_app.py
├── email_reader.py
├── smart_summarizer_v3.py      # deterministic baseline
├── priority_model.py
├── priority_tagging.py
├── feedback_system.py
├── context_loader.py
├── credentials_manager.py
├── requirements.txt
└── README.md
```

## Limitations

This project should not be presented as a production autonomous email agent. Important limitations include:

- LLM outputs can be incorrect or overconfident even when the schema is valid.
- Intent and urgency labels depend on the evaluation dataset and label definitions.
- User feedback can introduce preference bias.
- Local-model quality depends on the selected model and hardware.
- Hosted inference introduces external data-processing considerations.
- The current project does not establish broad generalisation across languages, organisations, or mailbox providers.

## Academic positioning

The strongest contribution of this repository is the **separation of concerns** between semantic LLM inference, deterministic baseline analysis, personalisation, feedback, and presentation. This makes the system easier to reproduce, evaluate, and extend than a monolithic API-driven demo.

For academic review, focus on the architecture, evaluation methodology, reproducibility, privacy assumptions, and limitations rather than claiming that an LLM automatically makes every inbox decision correctly.

## Future work

1. Build a labelled evaluation corpus with explicit intent and urgency definitions.
2. Compare at least one hosted and one local model under the same evaluation protocol.
3. Add multilingual evaluation, including German-language email samples.
4. Measure calibration and abstention behaviour for low-confidence analyses.
5. Add prompt/version tracking so experiments can be reproduced.
6. Integrate feedback into a documented learning protocol rather than silently modifying decisions.
7. Add data-minimisation and retention controls for real mailbox deployments.
