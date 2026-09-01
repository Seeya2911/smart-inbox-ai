"""
Finalize Action Generation Dataset

Final steps:
1. Duplicate detection and removal
2. Quality validation
3. Train/validation/test splitting
4. Generate comprehensive audit and statistics
5. Save final dataset files
"""

import pandas as pd
import json
import hashlib
import os
import difflib
from typing import Dict, List, Tuple, Any
from datetime import datetime
import random


class DatasetFinalizer:
    """Finalize and validate the action generation dataset."""

    def __init__(self, input_file: str):
        self.input_file = input_file
        self.dataset = []
        self.removed_examples = []
        self.audit_log = {
            "timestamp": datetime.now().isoformat(),
            "invalid_rows": [],
            "empty_bodies": [],
            "duplicates": [],
            "near_duplicates": [],
            "malformed_actions": [],
            "warnings": [],
            "quality_checks": {}
        }

    def load_data(self):
        """Load the annotated dataset with synthetic examples."""
        print(f"Loading dataset from {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} records")
        return len(self.dataset)

    def validate_quality(self):
        """Validate dataset quality."""
        print("\nValidating dataset quality...")

        invalid_count = 0
        empty_body_count = 0
        malformed_count = 0

        for i, record in enumerate(self.dataset):
            # Check for empty body
            if not record.get('body', '').strip():
                empty_body_count += 1
                self.audit_log['empty_bodies'].append({
                    "index": i,
                    "id": record.get('id'),
                    "subject": record.get('subject', '')[:50]
                })

            # Check for valid action type
            valid_actions = ["none", "reply", "create_task", "create_reminder",
                           "create_calendar_event", "review_document",
                           "contact_sender", "follow_up"]

            if record.get('action_type') not in valid_actions:
                malformed_count += 1
                self.audit_log['malformed_actions'].append({
                    "index": i,
                    "id": record.get('id'),
                    "action_type": record.get('action_type')
                })

            # Check for required fields when action is not "none"
            if record.get('action_type') != 'none':
                if not record.get('action_title'):
                    self.audit_log['warnings'].append({
                        "index": i,
                        "id": record.get('id'),
                        "issue": "Missing action_title for actionable email"
                    })

                if not record.get('source_evidence'):
                    self.audit_log['warnings'].append({
                        "index": i,
                        "id": record.get('id'),
                        "issue": "Missing source_evidence for actionable email"
                    })

        print(f"  Empty bodies: {empty_body_count}")
        print(f"  Malformed actions: {malformed_count}")
        print(f"  Warnings: {len(self.audit_log['warnings'])}")

        self.audit_log['quality_checks']['empty_body_count'] = empty_body_count
        self.audit_log['quality_checks']['malformed_count'] = malformed_count

    def detect_duplicates(self, similarity_threshold: float = 0.95):
        """Detect exact and near-duplicate emails."""
        print("\nDetecting duplicates...")

        duplicates = []
        near_duplicates = []

        # Create a mapping of normalized text to indices
        text_map = {}

        for i, record in enumerate(self.dataset):
            text = f"{record.get('subject', '')} {record.get('body', '')}".lower().strip()

            # Check for exact duplicates
            if text in text_map:
                duplicates.append((text_map[text], i))
                self.audit_log['duplicates'].append({
                    "index1": text_map[text],
                    "index2": i,
                    "id1": self.dataset[text_map[text]].get('id'),
                    "id2": record.get('id')
                })
            else:
                text_map[text] = i

        print(f"  Found {len(duplicates)} exact duplicates")

        # Check for near-duplicates (expensive, so use a smaller sample)
        # For full production, this would need a more sophisticated approach
        sample_size = min(len(self.dataset), 500)
        if len(self.dataset) > sample_size:
            print(f"  Sampling {sample_size} records for near-duplicate detection (performance optimization)...")
            indices = random.sample(range(len(self.dataset)), sample_size)
        else:
            indices = list(range(len(self.dataset)))

        # Only check pairs, not all combinations
        for idx, i in enumerate(indices):
            if idx % 100 == 0 and idx > 0:
                print(f"    Checked {idx}/{len(indices)} samples...")

            text_i = f"{self.dataset[i].get('subject', '')} {self.dataset[i].get('body', '')}".lower().strip()

            # Only compare with next 10 items to reduce O(n²) cost
            for j in indices[idx+1:min(idx+11, len(indices))]:
                text_j = f"{self.dataset[j].get('subject', '')} {self.dataset[j].get('body', '')}".lower().strip()

                # Quick length check before expensive similarity
                len_ratio = min(len(text_i), len(text_j)) / max(len(text_i), len(text_j))
                if len_ratio < 0.8:
                    continue

                similarity = difflib.SequenceMatcher(None, text_i, text_j).ratio()

                if similarity >= similarity_threshold and text_i != text_j:
                    near_duplicates.append((i, j, similarity))
                    self.audit_log['near_duplicates'].append({
                        "index1": i,
                        "index2": j,
                        "similarity": round(similarity, 3),
                        "id1": self.dataset[i].get('id'),
                        "id2": self.dataset[j].get('id')
                    })

        print(f"  Found {len(near_duplicates)} near-duplicates in sample (similarity >= {similarity_threshold})")

        return duplicates, near_duplicates

    def remove_exact_duplicates(self, duplicates: List[Tuple[int, int]]):
        """Remove exact duplicate entries (keep first occurrence)."""
        if not duplicates:
            print("\nNo exact duplicates to remove")
            return

        print(f"\nRemoving {len(duplicates)} exact duplicates...")

        # Collect indices to remove
        to_remove = set(idx2 for _, idx2 in duplicates)

        # Save removed examples
        for idx in sorted(to_remove, reverse=True):
            self.removed_examples.append({
                "reason": "exact_duplicate",
                "record": self.dataset[idx]
            })

        # Remove duplicates (iterate in reverse to maintain indices)
        for idx in sorted(to_remove, reverse=True):
            del self.dataset[idx]

        print(f"  Removed {len(to_remove)} duplicate records")
        print(f"  Remaining records: {len(self.dataset)}")

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Create stratified train/validation/test splits."""
        print("\nCreating train/validation/test splits...")

        # Stratify by action_type
        action_groups = {}
        for i, record in enumerate(self.dataset):
            action_type = record['action_type']
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(i)

        train_indices = []
        val_indices = []
        test_indices = []

        random.seed(42)

        for action_type, indices in action_groups.items():
            random.shuffle(indices)

            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        # Shuffle final splits
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        train_data = [self.dataset[i] for i in train_indices]
        val_data = [self.dataset[i] for i in val_indices]
        test_data = [self.dataset[i] for i in test_indices]

        print(f"  Train: {len(train_data)} ({len(train_data)/len(self.dataset)*100:.1f}%)")
        print(f"  Val:   {len(val_data)} ({len(val_data)/len(self.dataset)*100:.1f}%)")
        print(f"  Test:  {len(test_data)} ({len(test_data)/len(self.dataset)*100:.1f}%)")

        # Verify stratification
        print("\n  Action distribution verification:")
        for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            action_counts = {}
            for record in split_data:
                action_type = record['action_type']
                action_counts[action_type] = action_counts.get(action_type, 0) + 1

            print(f"    {split_name}:")
            for action_type, count in sorted(action_counts.items()):
                print(f"      {action_type:25s}: {count:4d}")

        return train_data, val_data, test_data

    def calculate_statistics(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Dict:
        """Calculate comprehensive dataset statistics."""
        print("\nCalculating statistics...")

        all_data = self.dataset

        stats = {
            "total_rows": len(all_data),
            "real_rows": sum(1 for x in all_data if not x.get('is_synthetic', False)),
            "synthetic_rows": sum(1 for x in all_data if x.get('is_synthetic', False)),
            "action_type_counts": {},
            "action_type_percentages": {},
            "none_count": 0,
            "none_percentage": 0.0,
            "duplicate_count": len(self.audit_log['duplicates']),
            "near_duplicate_count": len(self.audit_log['near_duplicates']),
            "removed_count": len(self.removed_examples),
            "examples_with_due_date": sum(1 for x in all_data if x.get('due_date')),
            "examples_with_due_time": sum(1 for x in all_data if x.get('due_time')),
            "examples_with_duration": sum(1 for x in all_data if x.get('duration_minutes')),
            "examples_with_participants": sum(1 for x in all_data if x.get('participants') and len(x['participants']) > 0),
            "train_count": len(train_data),
            "validation_count": len(val_data),
            "test_count": len(test_data),
            "dataset_hash": "",
            "creation_date": datetime.now().isoformat()
        }

        # Action type distribution
        for record in all_data:
            action_type = record['action_type']
            stats['action_type_counts'][action_type] = stats['action_type_counts'].get(action_type, 0) + 1

        # Calculate percentages
        if stats['total_rows'] > 0:
            for action_type, count in stats['action_type_counts'].items():
                stats['action_type_percentages'][action_type] = round(count / stats['total_rows'] * 100, 2)

            stats['none_count'] = stats['action_type_counts'].get('none', 0)
            stats['none_percentage'] = round(stats['none_count'] / stats['total_rows'] * 100, 2)

        # Calculate dataset hash
        dataset_str = json.dumps(all_data, sort_keys=True)
        stats['dataset_hash'] = hashlib.sha256(dataset_str.encode()).hexdigest()

        return stats

    def save_final_dataset(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], stats: Dict):
        """Save all final dataset files."""
        print("\nSaving final dataset files...")

        # Create directories
        os.makedirs('data/generation', exist_ok=True)
        os.makedirs('artifacts', exist_ok=True)

        # Save complete dataset (Excel)
        df_all = pd.DataFrame(self.dataset)
        excel_path = 'data/generation/smart_inbox_ai_action_dataset_v1.xlsx'
        df_all.to_excel(excel_path, index=False)
        print(f"  Saved: {excel_path}")

        # Save complete dataset (JSONL)
        jsonl_path = 'data/generation/smart_inbox_ai_action_dataset_v1.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for record in self.dataset:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  Saved: {jsonl_path}")

        # Save splits
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

        for split_name, split_data in splits.items():
            split_path = f'data/generation/smart_inbox_ai_action_dataset_v1_{split_name}.jsonl'
            with open(split_path, 'w', encoding='utf-8') as f:
                for record in split_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"  Saved: {split_path}")

        # Save statistics
        stats_path = 'artifacts/generation_action_dataset_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {stats_path}")

        # Save audit log
        audit_path = 'artifacts/generation_action_dataset_audit.json'
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(self.audit_log, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {audit_path}")

        # Save removed examples
        if self.removed_examples:
            removed_path = 'artifacts/generation_action_dataset_removed.jsonl'
            with open(removed_path, 'w', encoding='utf-8') as f:
                for record in self.removed_examples:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"  Saved: {removed_path}")

        return excel_path, jsonl_path

    def print_summary(self, stats: Dict):
        """Print final dataset summary."""
        print("\n" + "=" * 70)
        print("FINAL DATASET SUMMARY")
        print("=" * 70)
        print(f"\nTotal records:      {stats['total_rows']}")
        print(f"  Real examples:    {stats['real_rows']}")
        print(f"  Synthetic:        {stats['synthetic_rows']}")
        print(f"\nRemoved duplicates: {stats['removed_count']}")
        print(f"\nAction distribution:")
        for action_type, count in sorted(stats['action_type_counts'].items(), key=lambda x: x[1], reverse=True):
            pct = stats['action_type_percentages'][action_type]
            print(f"  {action_type:25s}: {count:5d} ({pct:5.1f}%)")
        print(f"\nNONE percentage:    {stats['none_percentage']:.1f}%")
        print(f"\nSplit sizes:")
        print(f"  Train:            {stats['train_count']}")
        print(f"  Validation:       {stats['validation_count']}")
        print(f"  Test:             {stats['test_count']}")
        print(f"\nStructured fields:")
        print(f"  With due_date:    {stats['examples_with_due_date']}")
        print(f"  With due_time:    {stats['examples_with_due_time']}")
        print(f"  With duration:    {stats['examples_with_duration']}")
        print(f"  With participants:{stats['examples_with_participants']}")
        print(f"\nDataset hash:       {stats['dataset_hash'][:16]}...")
        print(f"\nQuality checks:")
        print(f"  Duplicates found: {stats['duplicate_count']}")
        print(f"  Near-duplicates:  {stats['near_duplicate_count']}")
        print(f"  Warnings:         {len(self.audit_log['warnings'])}")
        print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("Finalizing Action Generation Dataset")
    print("=" * 70)

    # Initialize finalizer
    finalizer = DatasetFinalizer('data/generation/annotated_actions_with_synthetic.json')

    # Load data
    finalizer.load_data()

    # Validate quality
    finalizer.validate_quality()

    # Detect duplicates
    duplicates, near_duplicates = finalizer.detect_duplicates()

    # Remove exact duplicates
    finalizer.remove_exact_duplicates(duplicates)

    # Split dataset
    train_data, val_data, test_data = finalizer.split_dataset()

    # Calculate statistics
    stats = finalizer.calculate_statistics(train_data, val_data, test_data)

    # Save final dataset
    excel_path, jsonl_path = finalizer.save_final_dataset(train_data, val_data, test_data, stats)

    # Print summary
    finalizer.print_summary(stats)

    print("\n✓ Dataset creation complete!")
    print(f"\nMain files:")
    print(f"  {excel_path}")
    print(f"  {jsonl_path}")
    print(f"\nSplit files:")
    print(f"  data/generation/smart_inbox_ai_action_dataset_v1_train.jsonl")
    print(f"  data/generation/smart_inbox_ai_action_dataset_v1_validation.jsonl")
    print(f"  data/generation/smart_inbox_ai_action_dataset_v1_test.jsonl")

    return stats


if __name__ == "__main__":
    main()
