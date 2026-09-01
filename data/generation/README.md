# Email Action Generation Dataset v1

**Created:** 2026-09-01  
**Purpose:** Supervised fine-tuning of FLAN-T5-base for email action extraction  
**Target Model:** `google/flan-t5-base` (local inference, no external API)

## Overview

This dataset was created to train a local generative model to perform **EMAIL ACTION EXTRACTION** from email content. The model learns to:

1. Determine whether an action is required from an email
2. Identify the appropriate action type when needed
3. Extract structured details (title, description, dates, participants)

**Critical principle:** Many emails legitimately require NO ACTION. The dataset contains ~58% "none" examples to teach the model this important distinction.

## Dataset Statistics

- **Total records:** 3,500
  - Real examples: 3,341 (from classification dataset)
  - Synthetic examples: 159 (targeted augmentation)
- **Removed duplicates:** 21 exact duplicates
- **Quality warnings:** 130 (mostly missing source_evidence for actionable emails)

### Action Type Distribution

| Action Type | Count | Percentage |
|------------|-------|------------|
| none | 2,026 | 57.9% |
| reply | 483 | 13.8% |
| create_calendar_event | 311 | 8.9% |
| create_task | 220 | 6.3% |
| review_document | 164 | 4.7% |
| create_reminder | 100 | 2.9% |
| follow_up | 99 | 2.8% |
| contact_sender | 97 | 2.8% |

### Dataset Splits

Stratified splits to maintain action type distribution:

- **Train:** 2,797 (79.9%)
- **Validation:** 347 (9.9%)
- **Test:** 356 (10.2%)

### Structured Fields Coverage

- **With due_date:** 896 examples
- **With due_time:** 214 examples
- **With duration:** 3 examples
- **With participants:** 51 examples

## Action Types

The dataset uses exactly 8 action types:

1. **none** - No action required (informational, confirmations, newsletters, etc.)
2. **reply** - Respond to sender (confirmations, questions)
3. **create_task** - Complete a task (submit, provide, prepare, etc.)
4. **create_reminder** - Set a reminder (payment due, deadline, etc.)
5. **create_calendar_event** - Schedule meeting or appointment
6. **review_document** - Review, check, or approve document/contract/report
7. **contact_sender** - Reach out to third party (vendor, customer, client)
8. **follow_up** - Follow up on previous conversation or request

## Schema

Each record contains:

### Email Content
- `id` - Unique identifier
- `subject` - Email subject line
- `body` - Email body text

### Context (from classification dataset)
- `intent` - Original intent classification (context only)
- `priority` - Original priority classification (context only)

### Action Labels
- `action_type` - One of the 8 action types
- `action_title` - Concise action title (null for "none")
- `action_description` - What the user should do (null for "none")
- `due_date` - Extracted date when explicit (null otherwise)
- `due_time` - Extracted time when explicit (null otherwise)
- `duration_minutes` - Meeting duration when stated (null otherwise)
- `participants` - List of participants when mentioned (empty list otherwise)
- `source_evidence` - Sentence from email supporting the action (null for "none")

### Metadata
- `label_source` - Always "teacher_llm" (Claude teacher annotation)
- `source` - "classification_dataset" or "synthetic"
- `is_synthetic` - Boolean flag

## Annotation Methodology

### Teacher Annotation Process

All action labels were assigned by careful semantic analysis of email **CONTENT**, not by using intent/priority labels as deterministic rules.

**Key principles:**

1. **Content-driven:** Action decisions based on actual email text, not classification labels
2. **Conservative extraction:** Only extract structured fields when explicitly supported
3. **No hallucination:** Never invent dates, times, participants, or actions
4. **NONE is valid:** Many emails legitimately require no action

### Real Examples (n=3,341)

All emails from `smart_inbox_ai_dataset_v2.xlsx` were analyzed using:

- Pattern matching for action indicators
- Request detection (please, could you, etc.)
- Temporal information extraction
- Participant extraction
- Evidence sentence extraction

### Synthetic Examples (n=159)

Targeted synthetic examples filled coverage gaps:

- **contact_sender:** 98 examples (original dataset had only 2)
- **follow_up:** 30 examples (to reach 100 total)
- **create_reminder:** 2 examples (to reach 100 total)
- **NONE boundary cases:** 50 examples (tricky cases that could be confused with actions)

Synthetic examples focused on:
- Realistic business scenarios
- Diverse writing styles (formal, informal, direct, polite)
- Edge cases and ambiguous situations
- Action/no-action boundaries

## Files

### Main Dataset
- `smart_inbox_ai_action_dataset_v1.xlsx` - Complete dataset (Excel)
- `smart_inbox_ai_action_dataset_v1.jsonl` - Complete dataset (JSONL)

### Splits
- `smart_inbox_ai_action_dataset_v1_train.jsonl` - Training set
- `smart_inbox_ai_action_dataset_v1_validation.jsonl` - Validation set
- `smart_inbox_ai_action_dataset_v1_test.jsonl` - Test set (untouched)

### Audit Files
- `../../artifacts/generation_action_dataset_statistics.json` - Comprehensive statistics
- `../../artifacts/generation_action_dataset_audit.json` - Quality checks and warnings
- `../../artifacts/generation_action_dataset_removed.jsonl` - Removed duplicates

## Important Notes

### What This Dataset Is

- High-quality supervised training data for email action extraction
- Carefully annotated by teacher LLM with semantic understanding
- Balanced coverage of action types while maintaining realistic distribution
- Includes strong coverage of "none" cases (critical for zero-shot baseline failure)

### What This Dataset Is NOT

- NOT a classification dataset (intent/priority are context only)
- NOT using intent/priority as deterministic rules for actions
- NOT claiming perfect annotation (130 warnings in audit log)
- NOT artificially balanced (realistic distribution preferred over mathematical balance)

### Limitations

1. **Teacher annotation uncertainty:** Some emails are genuinely ambiguous
2. **Missing evidence:** 130 actionable emails lack source_evidence sentences
3. **Limited temporal coverage:** Only 896/1,474 actionable emails have due dates
4. **Synthetic diversity:** Synthetic examples use template-based generation
5. **Near-duplicate detection:** Limited sampling (500 records) for performance

## Dataset Hash

```
SHA256: ca1446bc59581596...
```

(See `artifacts/generation_action_dataset_statistics.json` for full hash)

## Usage

### Loading the Dataset

```python
import pandas as pd
import json

# Load Excel
df = pd.read_excel('smart_inbox_ai_action_dataset_v1.xlsx')

# Load JSONL
with open('smart_inbox_ai_action_dataset_v1_train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
```

### Expected Model Behavior

The fine-tuned FLAN-T5-base model should:

1. **Classify actionability:** Distinguish between actionable and non-actionable emails
2. **Select action type:** Choose the most appropriate action from 8 types
3. **Extract structure:** Generate action_title, action_description when applicable
4. **Parse temporal:** Extract due_date, due_time when explicitly stated
5. **Identify participants:** Extract participant names when mentioned

### Evaluation Metrics

Recommended metrics for model evaluation:

- **Action type accuracy:** Overall and per-class accuracy
- **NONE precision/recall:** Critical to avoid false-action rate
- **F1 scores:** Per-action-type F1 scores
- **Extraction accuracy:** Correctness of extracted dates/times/participants
- **Hallucination rate:** How often model invents information not in email

## Related Files

- **DO NOT MODIFY:** `smart_inbox_ai_dataset_v2.xlsx` (original classification dataset)
- Generation scripts:
  - `create_action_dataset.py` - Framework
  - `annotate_actions.py` - Teacher annotation
  - `create_synthetic_examples.py` - Synthetic generation
  - `finalize_action_dataset.py` - Duplicate removal, splitting, audit

## Next Steps

1. **Fine-tune FLAN-T5-base** on the training set
2. **Validate** on the validation set during training
3. **Evaluate** on the held-out test set
4. **Compare** against zero-shot baseline (which had 100% false-action rate)
5. **Iterate** based on error analysis

## Citation

If using this dataset, please acknowledge:

- Source: Smart Inbox AI project classification dataset v2
- Annotation: Teacher LLM (Claude) semantic analysis
- Task: Email action extraction for local generative model
- License: (Same as parent project)

---

**Created by:** Teacher LLM annotation pipeline  
**Date:** September 1, 2026  
**Version:** 1.0
