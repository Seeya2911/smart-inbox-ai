# Dataset provenance and training strategy

## Goal

Smart Inbox AI needs a training corpus that supports semantic email intent classification without overstating what public datasets can prove. The training data and final evaluation data are therefore treated as separate resources.

## Findings

### 1. Public email-intent data

The manually verified Enron intent dataset is a useful email-specific resource. It is derived from the Enron corpus and contains 3,655 manually verified email sentences labelled for action/response or significant information such as future events and deadlines. The companion repository includes a license file. It is suitable for learning email speech-act signals, but it is sentence-level, English-only, and does not directly provide the Smart Inbox urgency/priority taxonomy.

### 2. MASSIVE

Amazon's MASSIVE dataset contains more than one million utterances across 51 languages, with 60 intents and 55 slot types. It includes an `email` scenario and is licensed CC BY 4.0. It is valuable for multilingual intent representation and transfer learning, but it is a spoken-assistant NLU corpus rather than a corpus of natural email messages. It must therefore not be described as an email benchmark.

### 3. BANKING77

BANKING77 contains 13,083 English customer-service queries and 77 fine-grained banking intents under CC BY 4.0. It is useful as an auxiliary intent-classification benchmark or transfer-learning source, but its single banking domain makes it inappropriate as the primary Smart Inbox dataset.

### 4. Synthetic email-intent datasets

Several Hugging Face datasets contain synthetic email examples. These can be useful for development and controlled augmentation, but they should not be presented as real-world email evidence. In particular, small synthetic datasets are not sufficient justification for broad generalisation claims.

## Decision

The project will use a **multi-source training strategy**:

1. Use an email-specific public corpus where licensing and provenance are clear.
2. Use selected multilingual intent data such as MASSIVE only for auxiliary representation/transfer learning where the label semantics are defensible.
3. Add a clearly labelled synthetic training component only for gaps such as multilingual urgency, negation, implicit deadlines, and Smart Inbox-specific intent categories.
4. Keep the current multilingual evaluation corpus strictly out of training.
5. Create a separate manually reviewed gold evaluation set for Smart Inbox-specific claims.

No source will be silently remapped into a Smart Inbox label when the original semantics do not support that mapping. Any mapping will be documented and counted in the dataset manifest.

## Label policy

The primary task is multi-task classification:

- `intent`
- `urgency`
- `priority`

The model must not infer urgency from a single keyword. Training examples should contain explicit urgency, implicit deadlines, negation, polite/indirect requests, and context-dependent urgency.

## Split policy

Train/validation/test splits must be created before model fitting and must avoid near-duplicate or translated-pair leakage across splits. When a source contains parallel multilingual examples, all language variants of the same underlying utterance must remain in the same split.

The final gold test set must never be used for training, hyperparameter tuning, prompt development, or threshold selection.

## Claims policy

The repository will report source-specific results separately from the Smart Inbox gold evaluation. Results from banking, voice-assistant, or synthetic datasets must not be presented as evidence that the model understands unrestricted real-world email.

## Candidate sources

- Enron intent dataset (manually verified): email-specific, English, sentence-level.
- Amazon MASSIVE: multilingual NLU, CC BY 4.0, not an email corpus.
- PolyAI BANKING77: 13,083 English queries, 77 banking intents, CC BY 4.0, single-domain.

These sources are inputs to an evidence-based dataset design, not interchangeable benchmarks.
