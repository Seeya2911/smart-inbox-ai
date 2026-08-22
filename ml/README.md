# Supervised Multilingual INTENT Classification Pipeline

Smart Inbox AI provides a reproducible, supervised machine learning pipeline for the **INTENT** classification task, sitting alongside the legacy keyword baseline and optional LLM providers.

## Methodological Rigor & Guidelines

### 1. Evaluation Fallback Policy (No Test-Data Fitting)
- **Trainable models MUST receive a genuine training split.**
- Evaluators (`ml.evaluate_intent`) **MUST NEVER** fit `TfidfIntentClassifier` or `EmbeddingIntentClassifier` on test or evaluation data. Falling back to fitting trainable models on evaluation examples when no training split is provided is **methodologically invalid** and is **strictly prohibited**.
- If no training split or pre-trained model artifact is provided, evaluation of trainable models **FAILS LOUDLY (`ValueError`)**.
- The legacy **Keyword baseline** operates directly on test data because it consists of fixed deterministic rules without learned parameters.

### 2. Group-Aware Stratified Splitting Algorithm
- **Group Isolation Guarantee**: All examples sharing a `source_group_id` (or source file / thread ID) **MUST remain in exactly ONE split**. No `source_group_id` may appear in more than one of `train`, `validation`, or `test`. This prevents severe data leakage across augmented, translated, or thread-related examples.
- **Stratification Algorithm**: Uses a deterministic greedy bin-packing algorithm (`seed=42` by default). Groups are ordered by class-rarity priority, and allocated to splits to balance class distributions according to requested split ratios (`train_ratio=0.70`, `val_ratio=0.15`, `test_ratio=0.15`).
- **Strict Coverage Validation**: If a dataset contains too few groups per class to achieve group isolation while preserving class representation across splits, the pipeline **FAILS CLEARLY (`ValueError`)** rather than silently creating an invalid single-class or unstratified benchmark.

### 3. Development Fixtures vs. Real Benchmarks
- `tests/fixtures/intent_sample.jsonl` is a compact, deterministic development fixture created **exclusively for verifying pipeline mechanics and CI unit tests**.
- **Not a Benchmark**: Fixture metrics MUST NOT be presented as real-world model performance claims.
- **Provenance Cleanliness**: Source-derived examples retain their original source metadata and explicit label mappings. Synthetic dev examples are explicitly tagged with `is_synthetic: true`, `source_dataset: "synthetic_dev_fixture"`, and `provenance: "synthetic_development_only"`.

### 4. Pretrained Representation Model Attribution
- **Encoder**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (Apache-2.0, supporting 50+ languages, producing 768-dimensional sentence embeddings).
- **Attribution**: The foundation model is used **strictly as a frozen pretrained representation layer**. We do **NOT** claim that the foundation model itself was created or trained by this project.
- **Project-Trained Downstream Classifier**: This project trains and evaluates downstream task-specific classifiers (scikit-learn `LogisticRegression`) on top of generated embeddings. The foundation model weights are not fine-tuned in this stage.

### 5. Task Scope & Performance Claims Policy
- **Scope Limit**: **Urgency and Priority** are intentionally **NOT part of this first INTENT training experiment**. This pipeline focuses exclusively on reproducible intent classification.
- **No Unverified Performance Claims**: No performance claims or benchmark numbers should be added until the model is trained and evaluated on a sufficiently large, defensible, real-world corpus.

## Installation & Environment Setup

Install optional ML dependencies:

```bash
python -m pip install -r requirements-ml.txt
```

## Reproducible Pipeline CLIs

### 1. Training Pipeline CLI (`ml.train_intent`)

Train the downstream intent classifier with defensible label mapping, quality checks, group-aware deterministic splitting, and split leakage verification:

```bash
python -m ml.train_intent \
  --data tests/fixtures/intent_sample.jsonl \
  --classifier embedding \
  --output-model artifacts/intent_model.joblib \
  --output-dataset artifacts/intent_canonical_dataset.json \
  --seed 42
```

Pipeline Execution Steps:
1. **Ingestion & Provenance**: Preserves source IDs, original splits, and synthetic/source provenance.
2. **Defensible Label Mapping**: Maps valid labels into canonical taxonomy (`request`, `question`, `meeting`, `notification`, `promotion`, `complaint`, `follow_up`, `information`, `other`). Excludes unmappable labels and logs exclusion reasons.
3. **Data Quality Checks**: Enforces non-empty text, checks source ID duplicates, exact text duplicates, and label conflicts.
4. **Deterministic Group-Aware Stratified Splitting**: Splits data into train (70%), validation (15%), and test (15%) splits while preserving group isolation and class coverage.
5. **Pre-Training Leakage Verification**: Performs loud leakage detection (exact text, normalized text, source-group overlap, and near-duplicate Jaccard similarity across splits). The pipeline **fails loudly (`DataLeakageError`)** if leakage is detected.
6. **Model Fitting**: Extracts 768-dimensional embeddings and fits a downstream `LogisticRegression` classifier **strictly on the training split**.
7. **Artifact Export**: Saves model weights along with complete reproducibility metadata (seed, model ID, Python/package versions, configuration, timestamps, class/language distributions).

### 2. Evaluation CLI (`ml.evaluate_intent`)

Compare intent model architectures on the canonical dataset splits:

```bash
python -m ml.evaluate_intent \
  --data artifacts/intent_canonical_dataset.json \
  --model artifacts/intent_model.joblib
```

Evaluates and compares:
1. **Keyword Baseline** (Legacy deterministic rules)
2. **TF-IDF + Logistic Regression** (Classical NLP baseline fit on train split)
3. **Multilingual Transformer Embeddings + Logistic Regression** (Pretrained encoder + project-trained classifier fit on train split)

*Note: OpenAI API comparison is intentionally excluded from this offline ML experiment.*

### Metrics & Reporting Policy

Evaluation reports:
- Accuracy
- Macro F1
- Weighted F1
- Per-class Precision, Recall, F1, and Support
- Confusion Matrix
- Example counts, Class distribution, Language distribution

**Reporting Rule**: No metric is reported for classes that have 0 examples in the test split. If a test split contains only a single class, macro/weighted F1 metrics reflect single-class accuracy and an explicit disclaimer is attached.
