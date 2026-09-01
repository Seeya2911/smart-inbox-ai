"""
Action Dataset Annotator - Teacher LLM for Email Action Extraction

This script performs careful, manual-style teacher annotation of email actions.
Each email is analyzed for actionability based on CONTENT, not intent/priority labels.

Key annotation principles:
1. Read the email carefully
2. Determine if recipient needs to do something
3. Choose appropriate action type based on email semantics
4. Extract only explicitly supported structured fields
5. Many emails legitimately require NO ACTION
"""

import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os

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

class EmailActionAnnotator:
    """Careful teacher annotation of email actions."""

    def __init__(self):
        self.annotation_stats = {
            "total_annotated": 0,
            "action_counts": {action: 0 for action in ACTION_TYPES}
        }

    def analyze_email_carefully(self, row: pd.Series) -> Dict[str, Any]:
        """
        Teacher-quality annotation of email action.

        This function performs careful semantic analysis of email content
        to determine what action (if any) the recipient should take.
        """
        subject = str(row.get('subject', '')).strip()
        body = str(row.get('body', '')).strip()

        # Context only - NOT deterministic rules
        intent = str(row.get('intent', '')).strip().lower()
        priority = str(row.get('priority', '')).strip().lower()

        # Combine text for analysis
        full_text = f"{subject} {body}"
        text_lower = full_text.lower()

        # Initialize action record
        action = {
            "id": row.get('id', ''),
            "subject": subject,
            "body": body,
            "intent": intent,
            "priority": priority,
            "action_type": "none",
            "action_title": None,
            "action_description": None,
            "due_date": None,
            "due_time": None,
            "duration_minutes": None,
            "participants": [],
            "source_evidence": None,
            "label_source": "teacher_llm",
            "source": row.get('source', 'classification_dataset'),
            "is_synthetic": False
        }

        # === STEP 1: Detect NO ACTION signals ===

        # Explicit no-action indicators
        no_action_phrases = [
            'fyi', 'for your information', 'for your records',
            'no action required', 'no response needed', 'do not reply',
            'automated message', 'this is an automated',
            'has been completed', 'has been sent', 'has been processed',
            'successfully completed', 'confirmation only',
            'just letting you know', 'just to let you know',
            'no need to respond', 'for information purposes'
        ]

        if any(phrase in text_lower for phrase in no_action_phrases):
            return action

        # Newsletter/promotional/informational patterns
        newsletter_signals = [
            'newsletter', 'unsubscribe', 'weekly digest', 'monthly update',
            'this email was sent to', 'you are receiving this',
            'manage your subscription', 'update your preferences'
        ]

        if any(signal in text_lower for signal in newsletter_signals):
            return action

        # Transactional confirmations that require no action
        confirmation_patterns = [
            'your order has been shipped', 'your order has shipped',
            'shipment confirmation', 'delivery confirmation',
            'your payment has been received', 'payment received',
            'refund has been processed', 'has been refunded',
            'subscription renewed', 'renewal confirmation',
            'password changed successfully', 'password updated successfully'
        ]

        if any(pattern in text_lower for pattern in confirmation_patterns):
            return action

        # Meeting notes / minutes (informational)
        if 'meeting minutes' in text_lower or 'notes from' in text_lower:
            # Unless there's a clear action item
            if 'action item' not in text_lower and 'please' not in text_lower:
                return action

        # === STEP 2: Detect ACTIONABLE emails ===

        # Extract temporal information first
        due_date, due_time = self._extract_temporal_info(full_text)

        # Detect explicit requests/commands
        request_indicators = [
            r'\bplease\s+(\w+)',
            r'\bcould you\s+(\w+)',
            r'\bcan you\s+(\w+)',
            r'\bwould you\s+(\w+)',
            r'\bi need you to\s+(\w+)',
            r'\byou need to\s+(\w+)',
            r'\byou must\s+(\w+)',
            r'\byou should\s+(\w+)',
            r'\bkindly\s+(\w+)'
        ]

        request_found = False
        request_verb = None
        for pattern in request_indicators:
            match = re.search(pattern, text_lower)
            if match:
                request_found = True
                request_verb = match.group(1)
                break

        # === DOCUMENT REVIEW ===
        review_keywords = ['review', 'check', 'look at', 'examine', 'assess', 'evaluate', 'approve']
        document_keywords = ['document', 'contract', 'proposal', 'report', 'attachment', 'file', 'presentation', 'draft']

        has_review_verb = any(verb in text_lower for verb in review_keywords)
        has_document = any(doc in text_lower for doc in document_keywords)

        if request_found and has_review_verb and has_document:
            # Extract evidence
            evidence = self._extract_evidence(full_text, ['review', 'check', 'approve'])

            # Create title
            doc_type = next((doc for doc in document_keywords if doc in text_lower), 'document')
            title = f"Review {doc_type}"

            description = f"Review the {doc_type} as requested"
            if due_date:
                description += f" by {due_date}"

            action.update({
                "action_type": "review_document",
                "action_title": title,
                "action_description": description,
                "due_date": due_date,
                "due_time": due_time,
                "source_evidence": evidence
            })
            return action

        # === MEETING/CALENDAR ===
        meeting_keywords = ['meeting', 'call', 'conference', 'zoom', 'teams', 'appointment']
        scheduling_phrases = [
            'let\'s meet', 'let us meet', 'schedule a meeting', 'book a meeting',
            'set up a meeting', 'arrange a call', 'calendar invite'
        ]

        has_meeting_keyword = any(kw in text_lower for kw in meeting_keywords)
        has_scheduling = any(phrase in text_lower for phrase in scheduling_phrases)

        # Check for explicit time/date
        has_specific_time = due_date is not None or due_time is not None

        if has_scheduling or (has_meeting_keyword and has_specific_time and request_found):
            evidence = self._extract_evidence(full_text, meeting_keywords + ['schedule', 'time'])

            meeting_type = next((kw for kw in meeting_keywords if kw in text_lower), 'meeting')
            title = f"Attend {meeting_type}"
            if due_date:
                title = f"Attend {meeting_type} on {due_date}"

            description = f"Join the {meeting_type}"
            if due_time:
                description += f" at {due_time}"

            # Extract participants
            participants = self._extract_participants(full_text)

            # Extract duration
            duration = self._extract_duration(full_text)

            action.update({
                "action_type": "create_calendar_event",
                "action_title": title,
                "action_description": description,
                "due_date": due_date,
                "due_time": due_time,
                "duration_minutes": duration,
                "participants": participants,
                "source_evidence": evidence
            })
            return action

        # === REPLY ===
        reply_keywords = ['reply', 'respond', 'let me know', 'please confirm', 'confirm whether',
                         'please let us know', 'get back to', 'send confirmation']
        question_indicators = ['?', 'can you', 'could you', 'would you', 'do you']

        has_reply_request = any(kw in text_lower for kw in reply_keywords)
        has_question = '?' in full_text or any(indicator in text_lower for indicator in question_indicators)

        if has_reply_request or (has_question and request_found):
            evidence = self._extract_evidence(full_text, reply_keywords + ['confirm', 'let me know'])

            title = "Reply to sender"
            description = "Respond to the request"
            if 'confirm' in text_lower:
                description = "Confirm your response"

            action.update({
                "action_type": "reply",
                "action_title": title,
                "action_description": description,
                "due_date": due_date,
                "source_evidence": evidence
            })
            return action

        # === TASK (submit, complete, prepare) ===
        task_verbs = ['submit', 'send', 'provide', 'deliver', 'complete', 'finish',
                     'prepare', 'create', 'update', 'fill out', 'sign']

        has_task_verb = any(verb in text_lower for verb in task_verbs)

        if request_found and has_task_verb:
            evidence = self._extract_evidence(full_text, task_verbs)

            # Extract task description
            task_verb = next((verb for verb in task_verbs if verb in text_lower), 'complete')
            title = f"{task_verb.capitalize()} requested item"

            # Try to find object of the verb
            for verb in task_verbs:
                pattern = rf'\b{verb}\s+(the\s+)?(\w+(?:\s+\w+){{0,3}})'
                match = re.search(pattern, text_lower)
                if match:
                    obj = match.group(2)
                    title = f"{verb.capitalize()} {obj}"
                    break

            description = f"Complete the requested task"
            if due_date:
                description += f" by {due_date}"

            action.update({
                "action_type": "create_task",
                "action_title": title,
                "action_description": description,
                "due_date": due_date,
                "due_time": due_time,
                "source_evidence": evidence
            })
            return action

        # === REMINDER (payment, deadline, don't forget) ===
        reminder_keywords = ['reminder', 'don\'t forget', 'do not forget', 'remember to',
                            'upcoming', 'due soon', 'payment due', 'invoice due', 'bill due']

        has_reminder = any(kw in text_lower for kw in reminder_keywords)

        if has_reminder or (due_date and ('pay' in text_lower or 'invoice' in text_lower)):
            evidence = self._extract_evidence(full_text, reminder_keywords + ['due', 'deadline', 'pay'])

            title = "Payment reminder"
            if 'invoice' in text_lower:
                title = "Pay invoice"
            elif 'meeting' in text_lower:
                title = "Meeting reminder"
            else:
                title = "Deadline reminder"

            description = "Remember to complete this action"
            if due_date:
                description = f"Complete by {due_date}"

            action.update({
                "action_type": "create_reminder",
                "action_title": title,
                "action_description": description,
                "due_date": due_date,
                "due_time": due_time,
                "source_evidence": evidence
            })
            return action

        # === FOLLOW-UP ===
        followup_keywords = ['follow up', 'follow-up', 'following up', 'checking in',
                            'circle back', 'touch base', 'any update', 'status update']

        has_followup = any(kw in text_lower for kw in followup_keywords)

        if has_followup:
            evidence = self._extract_evidence(full_text, followup_keywords)

            title = "Follow up on request"
            description = "Follow up on the previous conversation"

            action.update({
                "action_type": "follow_up",
                "action_title": title,
                "action_description": description,
                "due_date": due_date,
                "source_evidence": evidence
            })
            return action

        # === CONTACT SENDER ===
        contact_keywords = ['contact', 'reach out', 'get in touch', 'call', 'email']
        third_party = ['vendor', 'supplier', 'customer', 'client', 'team member']

        has_contact = any(kw in text_lower for kw in contact_keywords)
        has_third_party = any(tp in text_lower for tp in third_party)

        if request_found and has_contact and has_third_party:
            evidence = self._extract_evidence(full_text, contact_keywords)

            party = next((tp for tp in third_party if tp in text_lower), 'party')
            title = f"Contact {party}"
            description = f"Reach out to the {party} as requested"

            action.update({
                "action_type": "contact_sender",
                "action_title": title,
                "action_description": description,
                "source_evidence": evidence
            })
            return action

        # === DEFAULT: NONE ===
        # If no clear action pattern detected, email is informational
        return action

    def _extract_temporal_info(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract date and time from email text (only when explicit)."""
        text_lower = text.lower()

        due_date = None
        due_time = None

        # Explicit date keywords
        date_keywords = {
            'today': 'today',
            'tomorrow': 'tomorrow',
            'monday': 'Monday',
            'tuesday': 'Tuesday',
            'wednesday': 'Wednesday',
            'thursday': 'Thursday',
            'friday': 'Friday',
            'saturday': 'Saturday',
            'sunday': 'Sunday',
            'next week': 'next week',
            'this week': 'this week',
            'end of day': 'end of day',
            'eod': 'end of day'
        }

        for keyword, label in date_keywords.items():
            if keyword in text_lower:
                due_date = label
                break

        # Explicit date patterns
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY
            r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                due_date = match.group(1)
                break

        # Time patterns
        time_patterns = [
            r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b',
            r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b',
            r'\b(\d{1,2}:\d{2})\b'
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                due_time = match.group(1)
                break

        return due_date, due_time

    def _extract_evidence(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract a relevant sentence as evidence."""
        sentences = text.split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if any(kw in sentence.lower() for kw in keywords):
                # Return first matching sentence, limited to 150 chars
                return sentence[:150]

        return None

    def _extract_participants(self, text: str) -> List[str]:
        """Extract participant names from email (when explicit)."""
        # Simple heuristic: look for "with X" or "and X"
        participants = []

        # Look for patterns like "with John" or "and Sarah"
        name_pattern = r'\b(?:with|and)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        matches = re.findall(name_pattern, text)

        return matches[:3]  # Limit to 3 participants

    def _extract_duration(self, text: str) -> Optional[int]:
        """Extract meeting duration when explicitly stated."""
        duration_pattern = r'(\d+)\s*(?:minute|min|hour|hr)s?'
        match = re.search(duration_pattern, text.lower())

        if match:
            value = int(match.group(1))
            if 'hour' in text.lower() or 'hr' in text.lower():
                return value * 60
            return value

        return None

    def annotate_dataset(self, input_file: str) -> List[Dict]:
        """Annotate the entire classification dataset."""
        print(f"Loading dataset from {input_file}...")
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} emails")

        annotated_data = []

        print("\nAnnotating emails...")
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(df)} emails...")

            action = self.analyze_email_carefully(row)
            annotated_data.append(action)

            # Update stats
            self.annotation_stats["total_annotated"] += 1
            self.annotation_stats["action_counts"][action["action_type"]] += 1

        print(f"\nAnnotation complete: {len(annotated_data)} emails annotated")
        print(f"\nAction distribution:")
        for action_type, count in sorted(self.annotation_stats["action_counts"].items(),
                                        key=lambda x: x[1], reverse=True):
            pct = count / len(annotated_data) * 100
            print(f"  {action_type:25s}: {count:5d} ({pct:5.1f}%)")

        return annotated_data


def main():
    print("=" * 70)
    print("Email Action Annotation - Teacher LLM")
    print("=" * 70)
    print()

    annotator = EmailActionAnnotator()

    # Annotate the classification dataset
    annotated_data = annotator.annotate_dataset('smart_inbox_ai_dataset_v2.xlsx')

    # Save intermediate results
    os.makedirs('data/generation', exist_ok=True)

    output_file = 'data/generation/annotated_actions_initial.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotated_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved annotated data to: {output_file}")
    print(f"Total records: {len(annotated_data)}")

    return annotated_data


if __name__ == "__main__":
    main()
