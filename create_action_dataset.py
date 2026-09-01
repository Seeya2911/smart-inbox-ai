"""
Smart Inbox AI - Action Generation Dataset Creation

This script creates a high-quality supervised dataset for fine-tuning FLAN-T5-base
on email action extraction.

Key principles:
1. Action decisions MUST be based on EMAIL CONTENT, not intent/priority labels
2. Many emails require NO ACTION (action_type = "none")
3. Only extract structured fields when explicitly supported by email text
4. No hallucination of dates, times, participants, or actions
"""

import pandas as pd
import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import re
from collections import defaultdict, Counter
import difflib

# Action types
ACTION_TYPES = [
    "none",
    "reply",
    "create_task",
    "create_reminder",
    "create_calendar_event",
    "review_document",
    "contact_sender",
    "follow_up"
]

class ActionDatasetCreator:
    """Teacher annotator for action generation dataset creation."""

    def __init__(self, input_file: str):
        self.input_file = input_file
        self.df = None
        self.action_dataset = []
        self.removed_examples = []
        self.audit_log = {
            "invalid_rows": [],
            "empty_bodies": [],
            "duplicates": [],
            "near_duplicates": [],
            "hallucination_checks": [],
            "warnings": []
        }

    def load_classification_dataset(self):
        """Load the existing classification dataset (READ-ONLY)."""
        print(f"Loading classification dataset from {self.input_file}...")
        self.df = pd.read_excel(self.input_file)
        print(f"Loaded {len(self.df)} rows")
        print(f"Columns: {list(self.df.columns)}")
        return self.df

    def analyze_email_for_action(self, row: pd.Series) -> Dict[str, Any]:
        """
        Carefully analyze email content to determine required action.

        CRITICAL: This function must read the ACTUAL EMAIL CONTENT.
        Do NOT use intent/priority labels as deterministic rules.
        """
        subject = str(row.get('subject', '')).strip()
        body = str(row.get('body', '')).strip()
        intent = str(row.get('intent', '')).strip().lower()
        priority = str(row.get('priority', '')).strip().lower()

        # Combine for analysis
        text = f"{subject} {body}".lower()

        # Initialize action record
        action = {
            "action_type": "none",
            "action_title": None,
            "action_description": None,
            "due_date": None,
            "due_time": None,
            "duration_minutes": None,
            "participants": [],
            "source_evidence": None
        }

        # Pattern matching for action detection
        # IMPORTANT: These are heuristics to help teacher annotation
        # The final model must learn from the labeled examples

        # Check for explicit "no action needed" signals
        no_action_signals = [
            'fyi', 'for your information', 'just letting you know',
            'for your records', 'this is automated', 'do not reply',
            'no response needed', 'no action required',
            'has been completed', 'has been sent', 'has been processed',
            'successfully', 'confirmation only'
        ]

        if any(signal in text for signal in no_action_signals):
            return action

        # Detect meeting-related actions
        meeting_keywords = ['meeting', 'call', 'conference', 'zoom', 'teams', 'webinar']
        meeting_scheduling = ['let\'s meet', 'schedule', 'calendar invite', 'book a time', 'set up a meeting']

        has_meeting_keyword = any(kw in text for kw in meeting_keywords)
        has_scheduling = any(phrase in text for phrase in meeting_scheduling)

        # Detect action verbs
        action_verbs = {
            'review': ['review', 'check', 'look at', 'examine', 'assess'],
            'submit': ['submit', 'send', 'provide', 'deliver', 'upload'],
            'reply': ['reply', 'respond', 'let me know', 'confirm', 'please answer'],
            'complete': ['complete', 'finish', 'do', 'perform', 'execute'],
            'pay': ['pay', 'payment', 'invoice due', 'bill due'],
            'attend': ['attend', 'join', 'participate'],
            'prepare': ['prepare', 'get ready', 'make sure'],
            'follow_up': ['follow up', 'check back', 'circle back', 'touch base']
        }

        # Detect request patterns
        request_patterns = [
            r'please\s+\w+',
            r'could you\s+\w+',
            r'can you\s+\w+',
            r'would you\s+\w+',
            r'need you to\s+\w+',
            r'i need\s+\w+',
            r'you (should|must|need to)\s+\w+'
        ]

        has_request = any(re.search(pattern, text) for pattern in request_patterns)

        # This is a SIMPLIFIED teacher heuristic
        # Real annotation requires reading full context

        # The actual implementation will analyze each email carefully
        # This is just the framework

        return action

    def extract_temporal_info(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract date and time information from email text.

        CRITICAL: Only extract when explicitly stated.
        Do NOT invent dates from vague expressions.
        """
        text_lower = text.lower()

        due_date = None
        due_time = None

        # Date patterns
        date_patterns = {
            'today': 'today',
            'tomorrow': 'tomorrow',
            'monday': 'next Monday',
            'tuesday': 'next Tuesday',
            'wednesday': 'next Wednesday',
            'thursday': 'next Thursday',
            'friday': 'next Friday',
            'saturday': 'next Saturday',
            'sunday': 'next Sunday',
        }

        # Look for explicit dates
        date_regex = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'
        date_match = re.search(date_regex, text)
        if date_match:
            due_date = date_match.group(1)
        else:
            for keyword, date_label in date_patterns.items():
                if keyword in text_lower:
                    due_date = date_label
                    break

        # Time patterns
        time_regex = r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|\d{1,2}\s*(?:AM|PM|am|pm))\b'
        time_match = re.search(time_regex, text)
        if time_match:
            due_time = time_match.group(1)

        return due_date, due_time

    def detect_duplicates(self, threshold: float = 0.95) -> List[Tuple[int, int, float]]:
        """Detect exact and near-duplicate emails."""
        duplicates = []

        print("Detecting duplicates...")

        for i in range(len(self.action_dataset)):
            for j in range(i + 1, len(self.action_dataset)):
                text_i = f"{self.action_dataset[i]['subject']} {self.action_dataset[i]['body']}"
                text_j = f"{self.action_dataset[j]['subject']} {self.action_dataset[j]['body']}"

                # Normalize
                text_i = text_i.lower().strip()
                text_j = text_j.lower().strip()

                # Check exact match
                if text_i == text_j:
                    duplicates.append((i, j, 1.0))
                    continue

                # Check similarity
                similarity = difflib.SequenceMatcher(None, text_i, text_j).ratio()
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))

        print(f"Found {len(duplicates)} duplicate/near-duplicate pairs")
        return duplicates

    def create_synthetic_examples(self, target_counts: Dict[str, int]) -> List[Dict]:
        """
        Create synthetic examples to fill coverage gaps.

        IMPORTANT: Only called after identifying genuine gaps.
        Examples must be realistic and diverse.
        """
        synthetic_examples = []

        # This will be implemented with careful diversity
        # Focus on underrepresented action types

        return synthetic_examples

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """Create train/validation/test splits."""
        import random

        # Shuffle
        random.seed(42)
        indices = list(range(len(self.action_dataset)))
        random.shuffle(indices)

        train_size = int(len(indices) * train_ratio)
        val_size = int(len(indices) * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return {
            'train': [self.action_dataset[i] for i in train_indices],
            'val': [self.action_dataset[i] for i in val_indices],
            'test': [self.action_dataset[i] for i in test_indices]
        }

    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive dataset statistics."""
        stats = {
            "total_rows": len(self.action_dataset),
            "real_rows": sum(1 for x in self.action_dataset if not x.get('is_synthetic', False)),
            "synthetic_rows": sum(1 for x in self.action_dataset if x.get('is_synthetic', False)),
            "action_type_counts": Counter(x['action_type'] for x in self.action_dataset),
            "action_type_percentages": {},
            "none_count": 0,
            "none_percentage": 0.0,
            "duplicate_count": len(self.audit_log.get('duplicates', [])),
            "near_duplicate_count": len(self.audit_log.get('near_duplicates', [])),
            "removed_count": len(self.removed_examples),
            "examples_with_due_date": sum(1 for x in self.action_dataset if x.get('due_date')),
            "examples_with_due_time": sum(1 for x in self.action_dataset if x.get('due_time')),
            "examples_with_duration": sum(1 for x in self.action_dataset if x.get('duration_minutes')),
            "examples_with_participants": sum(1 for x in self.action_dataset if x.get('participants')),
            "train_count": 0,
            "validation_count": 0,
            "test_count": 0,
            "dataset_hash": ""
        }

        # Calculate percentages
        if stats["total_rows"] > 0:
            for action_type, count in stats["action_type_counts"].items():
                stats["action_type_percentages"][action_type] = round(count / stats["total_rows"] * 100, 2)

            stats["none_count"] = stats["action_type_counts"].get("none", 0)
            stats["none_percentage"] = round(stats["none_count"] / stats["total_rows"] * 100, 2)

        return stats

    def save_dataset(self, output_dir: str):
        """Save the final dataset in multiple formats."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(output_dir), 'artifacts'), exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(self.action_dataset)

        # Save Excel
        excel_path = os.path.join(output_dir, 'smart_inbox_ai_action_dataset_v1.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Saved Excel dataset: {excel_path}")

        # Save JSONL
        jsonl_path = os.path.join(output_dir, 'smart_inbox_ai_action_dataset_v1.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for record in self.action_dataset:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved JSONL dataset: {jsonl_path}")

        # Save statistics
        stats = self.calculate_statistics()
        stats_path = 'artifacts/generation_action_dataset_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved statistics: {stats_path}")

        # Save audit log
        audit_path = 'artifacts/generation_action_dataset_audit.json'
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(self.audit_log, f, indent=2, ensure_ascii=False)
        print(f"Saved audit log: {audit_path}")

        # Save removed examples
        if self.removed_examples:
            removed_path = 'artifacts/generation_action_dataset_removed.jsonl'
            with open(removed_path, 'w', encoding='utf-8') as f:
                for record in self.removed_examples:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"Saved removed examples: {removed_path}")

        return excel_path, jsonl_path


def main():
    """Main execution function."""

    # Input dataset (READ-ONLY)
    input_file = 'smart_inbox_ai_dataset_v2.xlsx'
    output_dir = 'data/generation'

    print("=" * 60)
    print("Smart Inbox AI - Action Generation Dataset Creation")
    print("=" * 60)
    print()
    print("Task: Create supervised fine-tuning dataset for FLAN-T5-base")
    print("Target: Email action extraction")
    print()
    print("IMPORTANT PRINCIPLES:")
    print("  1. Action decisions based on EMAIL CONTENT")
    print("  2. Many emails require NO ACTION")
    print("  3. No hallucination of dates/times/participants")
    print("  4. Classification labels are CONTEXT ONLY")
    print()

    # Create dataset creator
    creator = ActionDatasetCreator(input_file)

    # Load classification dataset
    df = creator.load_classification_dataset()
    print(f"\nDataset preview:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:")
    print(df.dtypes)

    # The actual annotation will happen in the next step
    print("\n" + "=" * 60)
    print("Next: Careful teacher annotation of action labels")
    print("=" * 60)


if __name__ == "__main__":
    main()
