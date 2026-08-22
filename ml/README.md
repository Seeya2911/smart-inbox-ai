# Supervised Multilingual INTENT Classification Pipeline

Smart Inbox AI provides a reproducible, supervised machine learning pipeline for the **INTENT** classification task, sitting alongside the legacy keyword baseline and optional LLM providers.

## Pretrained Representation & Downstream Classifier

- **Pretrained Encoder**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Model Card & License**: Apache-2.0, supporting 50+ languages, outputting 768-dimensional sentence embeddings.
- **Foundation Model Attribution**: The foundation model is used **strictly as a frozen pretrained representation layer**. We do **NOT** claim that the foundation model itself was created or trained by this project.
- **Project-Trained Downstream Classifier**: This project trains and evaluates downstream task-specific classifiers (such as Logistic Regression) on top of the generated embeddings.

## Data Provenance & Corpus Distinctions

1. **Enron Intent Dataset**:
   - Source: Manually verified Enron intent dataset (`Charlie9/enron_intent_dataset_verified`).
   - Domain: **Email-specific source data**.
   - Defensible Label Mapping: Enron `ACTION_REQUIRED` maps defensibly to `request`. Binary `NO_ACTION_REQUIRED` labels are **excluded from supervised training** with documented reasons, as the absence of an action request does not map unambiguously to a single Smart Inbox intent class.

2. **Amazon Science MASSIVE Dataset**:
   - Domain: **Auxiliary multilingual NLU dataset** composed of voice-assistant utterances.
   - **Crucial Distinction**: MASSIVE is **NOT an email corpus** and must never be described or reported as one. It provides auxiliary NLU intent examples across English, German, French, and Spanish.

3. **Held-out Benchmark Policy**:
   - Do **NOT** train on `evaluation/multilingual_cases.jsonl`. That file is a compact, held-out evaluation corpus.
   - No benchmark results should be reported using training data.

4. **Task Scope**:
   - **Urgency and Priority** are intentionally **NOT part of this first INTENT training experiment**. This pipeline focuses exclusively on reproducible intent classification.

## Installation & Environment Setup

Install the optional ML dependencies:

```bash
python -m pip install -r requirements-ml.txt
```

## Reproducible Pipeline CLIs

### 1. Training Pipeline CLI (`ml.train_intent`)

Train the downstream intent classifier with defensible mapping, quality checks, deterministic splitting, and split leakage verification:

```bash
python -m ml.train_intent \
  --data tests/fixtures/intent_sample.jsonl \
  --classifier embedding \
  --output-model artifacts/intent_model.joblib \
  --output-dataset artifacts/intent_canonical_dataset.json \
  --seed 42
```

Pipeline Steps:
1. **Ingestion & Provenance**: Preserves original source IDs, original splits, and dataset provenance.
2. **Defensible Label Mapping**: Maps valid labels into canonical taxonomy (`request`, `question`, `meeting`, `notification`, `promotion`, `complaint`, `follow_up`, `information`, `other`). Excludes unmappable labels and logs exclusion reasons.
3. **Data Quality Checks**: Enforces non-empty text, checks source ID duplicates, exact text duplicates, and label conflicts.
4. **Deterministic Splitting**: Splits data into train (70%), validation (15%), and test (15%) using a fixed random seed and group-aware stratification.
5. **Leakage Verification**: Performs loud leakage detection (exact text, normalized text, and near-duplicate Jaccard similarity across splits). The pipeline **fails loudly (`DataLeakageError`)** if leakage is detected.
6. **Model Training**: Extracts 768-dimensional embeddings and trains a downstream `LogisticRegression` classifier.
7. **Artifact Export**: Saves model weights along with complete reproducibility metadata (seed, model ID, Python/package versions, configuration, timestamps, class/language distributions).

### 2. Evaluation CLI (`ml.evaluate_intent`)

Compare intent model architectures on the test split / evaluation corpus:

```bash
python -m ml.evaluate_intent \
  --data artifacts/intent_canonical_dataset.json \
  --model artifacts/intent_model.joblib
```

Evaluates and compares:
1. **Keyword Baseline** (Legacy deterministic rules)
2. **TF-IDF + Logistic Regression** (Classical NLP baseline)
3. **Multilingual Transformer Embeddings + Logistic Regression** (Pretrained encoder + project-trained classifier)

*Note: OpenAI API comparison is intentionally excluded from this offline ML experiment.*

### Metrics & Reporting Policy

Evaluation reports:
- Accuracy
- Macro F1
- Weighted F1
- Per-class Precision, Recall, F1, and Support
- Confusion Matrix
- Example counts, Class distribution, Language distribution

**Reporting Rule**: No metric is reported for classes that have 0 examples in the test split. No unverified performance claims or benchmark numbers should be added until full training experiments are executed on verified corpora.
