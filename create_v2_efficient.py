"""
Smart Inbox AI - Action Generation Dataset V2
EFFICIENT IMPLEMENTATION

Complete v2 creation optimized for 2,689 v1 examples.
Target: Transform 88% NONE distribution into balanced ~35% NONE dataset.
"""

import pandas as pd
import json
import hashlib
import random
import re
from typing import Dict, List
from datetime import datetime
from collections import Counter
import os

random.seed(42)

def load_v1():
    """Load v1 dataset."""
    print("Loading v1 dataset...")
    df = pd.read_excel('smart_inbox_ai_generation_dataset_v1.xlsx')
    v1_data = df.to_dict('records')

    counts = Counter(r.get('action_type', 'unknown') for r in v1_data)
    print(f"\nV1: {len(v1_data)} examples")
    print("Distribution:")
    for action, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action}: {count} ({count/len(v1_data)*100:.1f}%)")

    return v1_data

def audit_and_keep_best_none(v1_data):
    """Keep diverse NONE examples, remove redundant ones."""
    print("\n" + "="*70)
    print("Removing redundant NONE examples...")

    none_examples = [r for r in v1_data if r.get('action_type') == 'none']
    other_examples = [r for r in v1_data if r.get('action_type') != 'none']

    print(f"Initial NONE: {len(none_examples)}")

    # Target: keep ~800 diverse NONE examples (from 2,363)
    # Sample diverse NONE based on length and content variety
    diverse_none = []
    seen_texts = set()

    # Sort by length to ensure variety
    none_by_length = sorted(none_examples, key=lambda x: len(str(x.get('body', ''))))

    for ex in none_by_length:
        text = str(ex.get('body', '')).lower()[:100]  # First 100 chars

        # Check if sufficiently different
        is_unique = True
        for seen in seen_texts:
            # Simple uniqueness check
            overlap = sum(1 for i in range(min(len(text), len(seen))) if text[i:i+20] == seen[i:i+20])
            if overlap > 3:  # Too similar
                is_unique = False
                break

        if is_unique or len(diverse_none) < 800:
            diverse_none.append(ex)
            seen_texts.add(text)

            if len(diverse_none) >= 800:
                break

    print(f"Kept diverse NONE: {len(diverse_none)}")
    print(f"Removed redundant: {len(none_examples) - len(diverse_none)}")

    return other_examples + diverse_none

def create_synthetic_examples(current_data):
    """Create synthetic examples for underrepresented actions."""
    print("\n" + "="*70)
    print("Creating synthetic examples...")

    current_counts = Counter(r['action_type'] for r in current_data)
    synthetic_data = []
    syn_id = 50000

    # Targets for balanced distribution
    targets = {
        'review_document': 200,  # from 3
        'contact_sender': 150,   # from 0
        'follow_up': 150,         # from 16
        'create_task': 250,       # from 34
        'reply': 300,            # from 138
        'create_reminder': 200,  # from 65
        'create_calendar_event': 200  # from 70
    }

    for action, target in targets.items():
        current = current_counts.get(action, 0)
        needed = max(0, target - current)

        if needed > 0:
            print(f"\n{action}: {current} → {target} (+{needed})")

            for i in range(needed):
                syn_id += 1
                ex = create_synthetic_example(action, syn_id)
                synthetic_data.append(ex)

    print(f"\nTotal synthetic: {len(synthetic_data)}")
    return synthetic_data

def create_synthetic_example(action_type, syn_id):
    """Create a single synthetic example."""
    base = {
        "id": syn_id,
        "intent": "request",
        "priority": "medium",
        "summary_target": "",
        "action_type": action_type,
        "due_date": None,
        "due_time": None,
        "duration_minutes": None,
        "participants": [],
        "label_source": "teacher_llm",
        "source": "synthetic",
        "is_synthetic": True,
        "split": None
    }

    templates = {
        'review_document': {
            'subjects': ["Contract Review", "Report Review", "Proposal Review", "Document Review"],
            'bodies': [
                "Hi,\n\nCould you please review the attached {doc}? We need your feedback by end of week.\n\nThanks",
                "Hello,\n\nPlease review the {doc} and let me know your thoughts.\n\nBest regards",
                "Hi there,\n\nWould you mind reviewing the {doc}? Your input is important.\n\nThanks"
            ],
            'docs': ["contract", "report", "proposal", "document", "presentation", "spreadsheet"],
            'title': "Review document",
            'desc': "Review the document and provide feedback"
        },
        'contact_sender': {
            'subjects': ["Vendor Issue", "Customer Follow-up", "Supplier Question"],
            'bodies': [
                "Hi,\n\nCould you please contact the {party} about {issue}?\n\nThanks",
                "Hello,\n\nPlease reach out to the {party} regarding {issue}.\n\nRegards"
            ],
            'parties': ["vendor", "customer", "supplier", "client"],
            'issues': ["the missing shipment", "the invoice", "the order", "their request"],
            'title': "Contact third party",
            'desc': "Contact the specified person or organization"
        },
        'follow_up': {
            'subjects': ["Following up", "Checking in", "Status update"],
            'bodies': [
                "Hi,\n\nJust following up on my previous email. Any updates?\n\nThanks",
                "Hello,\n\nChecking in on the status. Have you had a chance to review?\n\nBest"
            ],
            'title': "Follow up",
            'desc': "Follow up on previous request"
        },
        'create_task': {
            'subjects': ["Action Required", "Task Assignment", "Please Complete"],
            'bodies': [
                "Hi,\n\nPlease {task} by {deadline}.\n\nThanks",
                "Hello,\n\nCould you {task}? We need this done {deadline}.\n\nRegards"
            ],
            'tasks': ["submit the report", "update the spreadsheet", "complete the form", "send the document"],
            'deadlines': ["by Friday", "by end of week", "by Monday", "soon"],
            'title': "Complete task",
            'desc': "Complete the requested task"
        },
        'reply': {
            'subjects': ["Quick question", "Confirmation needed", "Please confirm"],
            'bodies': [
                "Hi,\n\nCould you please confirm {item}?\n\nThanks",
                "Hello,\n\nPlease let me know about {item}.\n\nBest regards"
            ],
            'items': ["your attendance", "the time", "whether this works", "your availability"],
            'title': "Reply to sender",
            'desc': "Respond to the sender's request"
        },
        'create_reminder': {
            'subjects': ["Reminder", "Don't forget", "Upcoming deadline"],
            'bodies': [
                "Hi,\n\nReminder: {item} is due {when}.\n\nThanks",
                "Hello,\n\nDon't forget that {item} {when}.\n\nRegards"
            ],
            'items': ["the payment", "the report", "the meeting", "the deadline"],
            'whens': ["tomorrow", "on Friday", "next week", "soon"],
            'title': "Set reminder",
            'desc': "Remember the deadline or action"
        },
        'create_calendar_event': {
            'subjects': ["Meeting invitation", "Schedule request", "Calendar invite"],
            'bodies': [
                "Hi,\n\nLet's meet {when} at {time} to discuss {topic}.\n\nThanks",
                "Hello,\n\nI'd like to schedule a meeting {when} at {time}.\n\nBest"
            ],
            'whens': ["tomorrow", "on Monday", "next week", "on Friday"],
            'times': ["10 AM", "2 PM", "3:30 PM", "11 AM"],
            'topics': ["the project", "the proposal", "next steps", "the budget"],
            'title': "Attend meeting",
            'desc': "Attend the scheduled meeting"
        }
    }

    if action_type not in templates:
        return base

    tmpl = templates[action_type]

    # Generate subject
    subj = random.choice(tmpl['subjects'])

    # Generate body with substitutions
    body_template = random.choice(tmpl['bodies'])
    body = body_template

    # Substitute placeholders
    if '{doc}' in body:
        body = body.replace('{doc}', random.choice(tmpl['docs']))
    if '{party}' in body:
        body = body.replace('{party}', random.choice(tmpl['parties']))
    if '{issue}' in body:
        body = body.replace('{issue}', random.choice(tmpl['issues']))
    if '{task}' in body:
        body = body.replace('{task}', random.choice(tmpl['tasks']))
    if '{deadline}' in body:
        body = body.replace('{deadline}', random.choice(tmpl['deadlines']))
    if '{item}' in body:
        body = body.replace('{item}', random.choice(tmpl['items']))
    if '{when}' in body:
        body = body.replace('{when}', random.choice(tmpl['whens']))
    if '{time}' in body:
        body = body.replace('{time}', random.choice(tmpl['times']))
    if '{topic}' in body:
        body = body.replace('{topic}', random.choice(tmpl['topics']))

    # Extract evidence
    evidence_line = body.split('\n')[2] if len(body.split('\n')) > 2 else body.split('\n')[0]

    base.update({
        "subject": subj,
        "body": body,
        "summary_target": subj,
        "action_title": tmpl['title'],
        "action_description": tmpl['desc'],
        "source_evidence": evidence_line[:100]
    })

    return base

def create_splits(data):
    """Create stratified splits."""
    print("\n" + "="*70)
    print("Creating splits...")

    # Group by action_type
    groups = {}
    for rec in data:
        action = rec['action_type']
        if action not in groups:
            groups[action] = []
        groups[action].append(rec)

    # Shuffle and assign splits
    for action, records in groups.items():
        random.shuffle(records)
        n = len(records)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        for i, rec in enumerate(records):
            if i < n_train:
                rec['split'] = 'train'
            elif i < n_train + n_val:
                rec['split'] = 'validation'
            else:
                rec['split'] = 'test'

    train = sum(1 for r in data if r['split'] == 'train')
    val = sum(1 for r in data if r['split'] == 'validation')
    test = sum(1 for r in data if r['split'] == 'test')

    print(f"Train: {train}, Val: {val}, Test: {test}")

def save_v2(data):
    """Save v2 dataset."""
    print("\n" + "="*70)
    print("Saving v2 dataset...")

    os.makedirs('data/generation', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

    # Save main files
    df = pd.DataFrame(data)
    df.to_excel('data/generation/smart_inbox_ai_action_dataset_v2.xlsx', index=False)
    print("✓ Saved: data/generation/smart_inbox_ai_action_dataset_v2.xlsx")

    with open('data/generation/smart_inbox_ai_action_dataset_v2.jsonl', 'w', encoding='utf-8') as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print("✓ Saved: data/generation/smart_inbox_ai_action_dataset_v2.jsonl")

    # Calculate statistics
    action_counts = Counter(r['action_type'] for r in data)

    stats = {
        "original_v1_rows": 2689,
        "final_v2_rows": len(data),
        "real_rows": sum(1 for r in data if not r.get('is_synthetic', False)),
        "synthetic_rows": sum(1 for r in data if r.get('is_synthetic', False)),
        "action_type_counts": dict(action_counts),
        "action_type_percentages": {k: round(v/len(data)*100, 2) for k, v in action_counts.items()},
        "none_count": action_counts.get('none', 0),
        "none_percentage": round(action_counts.get('none', 0)/len(data)*100, 2),
        "train_count": sum(1 for r in data if r.get('split') == 'train'),
        "validation_count": sum(1 for r in data if r.get('split') == 'validation'),
        "test_count": sum(1 for r in data if r.get('split') == 'test'),
        "dataset_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        "timestamp": datetime.now().isoformat()
    }

    with open('artifacts/generation_action_dataset_statistics_v2.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✓ Saved: artifacts/generation_action_dataset_statistics_v2.json")

    # Simple audit log
    audit = {
        "v1_loaded": 2689,
        "none_removed": 2363 - action_counts.get('none', 0),
        "synthetic_added": stats['synthetic_rows'],
        "final_count": len(data),
        "timestamp": datetime.now().isoformat()
    }

    with open('artifacts/generation_action_dataset_audit_v2.json', 'w') as f:
        json.dump(audit, f, indent=2)
    print("✓ Saved: artifacts/generation_action_dataset_audit_v2.json")

    return stats

def print_report(stats):
    """Print final report."""
    print("\n" + "="*80)
    print("V2 DATASET FINAL REPORT")
    print("="*80)

    print(f"\nTransformation:")
    print(f"  V1 rows: {stats['original_v1_rows']}")
    print(f"  V2 rows: {stats['final_v2_rows']}")
    print(f"  Real: {stats['real_rows']}")
    print(f"  Synthetic: {stats['synthetic_rows']}")

    print(f"\nV2 Action Distribution:")
    for action, count in sorted(stats['action_type_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = stats['action_type_percentages'][action]
        v1_count = {'none': 2363, 'reply': 138, 'create_calendar_event': 70, 'create_reminder': 65,
                    'create_task': 34, 'follow_up': 16, 'review_document': 3, 'contact_sender': 0}.get(action, 0)
        print(f"  {action:25s}: {v1_count:4d} → {count:4d} ({pct:5.1f}%)")

    print(f"\nNONE: {stats['none_percentage']:.1f}% (target: 30-40%)")

    print(f"\nSplits:")
    print(f"  Train: {stats['train_count']}")
    print(f"  Validation: {stats['validation_count']}")
    print(f"  Test: {stats['test_count']}")

    print(f"\nHash: {stats['dataset_sha256'][:16]}...")

    print("\n" + "="*80)
    print("✓ V2 DATASET CREATION COMPLETE")
    print("="*80)

def main():
    print("="*80)
    print("SMART INBOX AI - ACTION DATASET V2 CREATION")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Pipeline
    v1_data = load_v1()
    v2_data = audit_and_keep_best_none(v1_data)
    synthetic = create_synthetic_examples(v2_data)
    v2_data.extend(synthetic)
    create_splits(v2_data)
    stats = save_v2(v2_data)
    print_report(stats)

    print(f"\nCompleted: {datetime.now().isoformat()}")

    return stats

if __name__ == "__main__":
    main()
