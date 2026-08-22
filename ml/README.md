# Multilingual ML classifier

Smart Inbox AI now has a local supervised ML path in addition to the legacy keyword baseline and optional LLM provider.

## Model design

The primary encoder is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. The public model card reports support for 50 languages and an Apache-2.0 license. It produces 768-dimensional sentence embeddings and is used here as a pretrained representation layer. The project trains its own task-specific Logistic Regression heads for intent, urgency, and priority. The pretrained foundation model is therefore not claimed as project-created; the downstream classifiers are trained by this project.

Model reference: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2

## Training data policy

Do **not** train on `evaluation/multilingual_cases.jsonl`. That file is a held-out evaluation corpus and is intentionally only 12 cases. The training CLI rejects very small corpora by default (`--min-cases 200`) so the benchmark cannot accidentally be presented as a meaningful training set.

The eventual training corpus should be a documented, licensed dataset or a documented mixture of public and clearly labelled synthetic data. Source provenance, licenses, language distribution, label mapping, and split methodology must be recorded before reporting model metrics.

## Training

Install the optional ML stack from the repository root:

```bash
python -m pip install -r requirements-ml.txt
```

Then provide a separate JSONL training corpus with:

```text
id, language, subject, body, intent, urgency, priority
```

Train with:

```bash
python -m ml.train_multilingual --train path/to/train.jsonl --output artifacts/multilingual_email_model.joblib
```

The script uses a fixed seed, stratified train/test split, duplicate-text detection, and reports accuracy and macro-F1 for each task. It refuses to train on a corpus smaller than 200 cases by default.

## Inference

```bash
python -m ml.predict_multilingual \
  --model artifacts/multilingual_email_model.joblib \
  --subject "Please review this by Friday" \
  --body "Could you send me the updated document before the deadline?"
```

## Research comparison

The intended experimental progression is:

1. transparent keyword baseline;
2. TF-IDF/classical ML baseline (to be added);
3. pretrained multilingual encoder + project-trained classification heads;
4. optional LLM comparison.

All systems should be evaluated on the same held-out multilingual benchmark. No model should be judged using training examples, and no API result should be presented as a model-training result.
