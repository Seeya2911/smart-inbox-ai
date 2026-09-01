"""
Synthetic Example Generation for Action Dataset

Creates targeted synthetic examples to fill coverage gaps identified in the
annotated dataset. Focus on:
1. Underrepresented action types (contact_sender, follow_up)
2. Action/no-action boundary cases
3. Diverse writing styles and scenarios
4. Realistic edge cases
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime


class SyntheticExampleGenerator:
    """Generate realistic synthetic examples for action dataset."""

    def __init__(self, existing_data: List[Dict]):
        self.existing_data = existing_data
        self.synthetic_id_counter = 10000  # Start synthetic IDs at 10000

    def analyze_gaps(self) -> Dict[str, int]:
        """Identify coverage gaps in the existing dataset."""
        action_counts = {}
        for record in self.existing_data:
            action_type = record['action_type']
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        print("\nCurrent action distribution:")
        total = len(self.existing_data)
        for action_type, count in sorted(action_counts.items(), key=lambda x: x[1]):
            pct = count / total * 100
            print(f"  {action_type:25s}: {count:5d} ({pct:5.1f}%)")

        # Calculate target counts for underrepresented actions
        # We want at least 100 examples per action type
        min_target = 100
        gaps = {}

        for action_type in ['contact_sender', 'follow_up', 'create_reminder', 'review_document']:
            current = action_counts.get(action_type, 0)
            if current < min_target:
                gaps[action_type] = min_target - current

        print("\nIdentified gaps:")
        for action_type, needed in gaps.items():
            print(f"  {action_type}: need {needed} more examples")

        return gaps

    def generate_contact_sender_examples(self, count: int) -> List[Dict]:
        """Generate contact_sender action examples."""
        examples = []

        templates = [
            {
                "subject": "Vendor Issue - {item}",
                "body": "Hi,\n\nWe're experiencing an issue with {item}. Could you please contact the vendor ({vendor}) to resolve this? {urgency}\n\nThanks,\n{sender}",
                "vendor": ["ABC Supplies", "TechCorp", "Global Services", "DataFlow Inc", "CloudVendor"],
                "item": ["the shipment", "the invoice", "the service outage", "the missing order", "the contract terms"],
                "urgency": ["This is urgent.", "Please do this by Friday.", "We need this resolved ASAP.", ""],
                "sender": ["Sarah", "Mike", "Jennifer", "David"]
            },
            {
                "subject": "Client Follow-up Needed",
                "body": "{greeting}\n\nCould you reach out to {client} regarding {matter}? They've been trying to get in touch and need a response by {deadline}.\n\n{closing}",
                "greeting": ["Hi", "Hello", "Good morning"],
                "client": ["our client at Acme Corp", "the customer from last week", "John at Enterprise Solutions", "the vendor we spoke with"],
                "matter": ["the project update", "their outstanding questions", "the contract renewal", "the support ticket"],
                "deadline": ["end of week", "tomorrow", "Friday", "next Monday"],
                "closing": ["Thanks", "Best regards", "Appreciate your help"]
            },
            {
                "subject": "Please Contact {party} About {issue}",
                "body": "Hi team,\n\n{party} needs to be contacted about {issue}. Can someone handle this? {details}\n\n{closing}",
                "party": ["the supplier", "our customer", "the IT vendor", "the contractor", "the client"],
                "issue": ["the delayed delivery", "billing discrepancies", "technical issues", "service renewal"],
                "details": ["Please get back to me once this is done.", "They're waiting for our response.", "This should be prioritized.", ""],
                "closing": ["Thanks", "Regards", "Best"]
            }
        ]

        for i in range(count):
            template = random.choice(templates)
            subject = template["subject"]
            body = template["body"]

            # Fill in variables
            for key, values in template.items():
                if key not in ["subject", "body"] and isinstance(values, list):
                    value = random.choice(values)
                    subject = subject.replace(f"{{{key}}}", value)
                    body = body.replace(f"{{{key}}}", value)

            examples.append({
                "id": self.synthetic_id_counter + i,
                "subject": subject,
                "body": body,
                "intent": "request",
                "priority": random.choice(["medium", "high"]),
                "action_type": "contact_sender",
                "action_title": "Contact third party",
                "action_description": "Reach out to the specified party to address the issue",
                "due_date": random.choice([None, "Friday", "end of week", None]),
                "due_time": None,
                "duration_minutes": None,
                "participants": [],
                "source_evidence": body.split('\n\n')[1][:100] if '\n\n' in body else body[:100],
                "label_source": "teacher_llm",
                "source": "synthetic",
                "is_synthetic": True
            })

        self.synthetic_id_counter += count
        return examples

    def generate_follow_up_examples(self, count: int) -> List[Dict]:
        """Generate follow_up action examples."""
        examples = []

        templates = [
            {
                "subject": "Following up on {topic}",
                "body": "Hi {name},\n\nJust following up on {topic} from {timeframe}. Have you had a chance to {action}?\n\nLet me know if you need anything.\n\n{closing}",
                "name": ["there", "team", ""],
                "topic": ["my previous email", "our conversation", "the proposal I sent", "the document request"],
                "timeframe": ["last week", "our meeting", "Monday", "earlier this week"],
                "action": ["review it", "take a look", "provide feedback", "give your thoughts"],
                "closing": ["Thanks", "Best", "Regards"]
            },
            {
                "subject": "Checking in - {matter}",
                "body": "Hello,\n\nI wanted to check in regarding {matter}. {question}\n\n{closing},\n{sender}",
                "matter": ["the proposal", "your decision", "the pending request", "the project status"],
                "question": ["Any updates?", "Have you had time to review?", "Where are we on this?", "Can you provide an update?"],
                "closing": ["Thanks", "Best regards", "Appreciate it"],
                "sender": ["Alex", "Chris", "Taylor", "Jordan"]
            },
            {
                "subject": "Re: {original_subject}",
                "body": "Hi,\n\nCircling back on this. {request}\n\nThanks!",
                "original_subject": ["Project Update", "Budget Approval", "Meeting Request", "Document Review"],
                "request": ["Did you get a chance to look at this?", "Any progress on this?", "Just wanted to follow up.", "Checking if you need anything from me."]
            }
        ]

        for i in range(count):
            template = random.choice(templates)
            subject = template["subject"]
            body = template["body"]

            for key, values in template.items():
                if key not in ["subject", "body"] and isinstance(values, list):
                    value = random.choice(values)
                    subject = subject.replace(f"{{{key}}}", value)
                    body = body.replace(f"{{{key}}}", value)

            examples.append({
                "id": self.synthetic_id_counter + i,
                "subject": subject,
                "body": body,
                "intent": "follow_up",
                "priority": random.choice(["low", "medium"]),
                "action_type": "follow_up",
                "action_title": "Follow up on previous request",
                "action_description": "Follow up on the previous conversation or request",
                "due_date": None,
                "due_time": None,
                "duration_minutes": None,
                "participants": [],
                "source_evidence": body.split('\n\n')[0][:100] if '\n\n' in body else body[:100],
                "label_source": "teacher_llm",
                "source": "synthetic",
                "is_synthetic": True
            })

        self.synthetic_id_counter += count
        return examples

    def generate_none_boundary_cases(self, count: int) -> List[Dict]:
        """Generate tricky NONE cases that might be confused with actions."""
        examples = []

        templates = [
            # Completed actions
            {
                "subject": "Task Completed - {task}",
                "body": "Hi,\n\nJust wanted to let you know that {task} has been completed. No further action needed on your part.\n\n{closing}",
                "task": ["the report", "your request", "the document review", "the payment processing"],
                "closing": ["Thanks", "Best", "FYI"]
            },
            # Automated confirmations
            {
                "subject": "Confirmation: {action}",
                "body": "This is an automated message.\n\nYour {action} has been successfully processed. This is for your records only.\n\nDo not reply to this email.",
                "action": ["payment", "subscription renewal", "order", "registration", "password change"]
            },
            # FYI messages
            {
                "subject": "FYI: {topic}",
                "body": "Hi team,\n\nFYI - {info}. No action required, just keeping you in the loop.\n\n{closing}",
                "info": ["the server maintenance is scheduled for tonight", "the meeting has been rescheduled", "the report has been published", "the project is on track"],
                "closing": ["Thanks", "Best", "Regards"]
            },
            # Status updates
            {
                "subject": "Status Update - {project}",
                "body": "{greeting},\n\nQuick update on {project}: {status}. Will keep you posted.\n\n{closing}",
                "greeting": ["Hi", "Hello", "Team"],
                "project": ["the migration", "Q3 planning", "the new feature", "the vendor contract"],
                "status": ["everything is proceeding as planned", "we're on schedule", "no issues to report", "completed successfully"],
                "closing": ["Thanks", "Best", "Talk soon"]
            }
        ]

        for i in range(count):
            template = random.choice(templates)
            subject = template["subject"]
            body = template["body"]

            for key, values in template.items():
                if key not in ["subject", "body"] and isinstance(values, list):
                    value = random.choice(values)
                    subject = subject.replace(f"{{{key}}}", value)
                    body = body.replace(f"{{{key}}}", value)

            examples.append({
                "id": self.synthetic_id_counter + i,
                "subject": subject,
                "body": body,
                "intent": random.choice(["notification", "information", "follow_up"]),
                "priority": random.choice(["low", "medium"]),
                "action_type": "none",
                "action_title": None,
                "action_description": None,
                "due_date": None,
                "due_time": None,
                "duration_minutes": None,
                "participants": [],
                "source_evidence": None,
                "label_source": "teacher_llm",
                "source": "synthetic",
                "is_synthetic": True
            })

        self.synthetic_id_counter += count
        return examples

    def generate_diverse_reminders(self, count: int) -> List[Dict]:
        """Generate diverse reminder examples."""
        examples = []

        scenarios = [
            {
                "subject": "Payment Reminder - Invoice #{invoice_num}",
                "body": "Dear customer,\n\nThis is a reminder that invoice #{invoice_num} for ${amount} is due on {date}.\n\nPlease process payment at your earliest convenience.\n\nThank you.",
                "invoice_num": lambda: random.randint(1000, 9999),
                "amount": lambda: random.randint(100, 5000),
                "date": ["Friday", "September 5", "end of month"]
            },
            {
                "subject": "Reminder: {event} {timeframe}",
                "body": "Hi,\n\nQuick reminder that {event} is {timeframe}. {extra}\n\nSee you then!",
                "event": ["our meeting", "the deadline", "the presentation", "your appointment"],
                "timeframe": ["tomorrow", "later today", "on Friday", "next week"],
                "extra": ["Don't forget to bring your materials.", "Please confirm your attendance.", "", ""]
            }
        ]

        for i in range(count):
            scenario = random.choice(scenarios)
            subject = scenario["subject"]
            body = scenario["body"]

            for key, value in scenario.items():
                if key not in ["subject", "body"]:
                    if callable(value):
                        replacement = str(value())
                    elif isinstance(value, list):
                        replacement = random.choice(value)
                    else:
                        replacement = str(value)

                    subject = subject.replace(f"{{{key}}}", replacement)
                    body = body.replace(f"{{{key}}}", replacement)

            examples.append({
                "id": self.synthetic_id_counter + i,
                "subject": subject,
                "body": body,
                "intent": "notification",
                "priority": random.choice(["medium", "high"]),
                "action_type": "create_reminder",
                "action_title": "Set reminder",
                "action_description": "Set a reminder for the upcoming deadline or event",
                "due_date": random.choice(["Friday", "tomorrow", "next week"]),
                "due_time": None,
                "duration_minutes": None,
                "participants": [],
                "source_evidence": body[:100],
                "label_source": "teacher_llm",
                "source": "synthetic",
                "is_synthetic": True
            })

        self.synthetic_id_counter += count
        return examples

    def generate_all_synthetic(self, gaps: Dict[str, int]) -> List[Dict]:
        """Generate all needed synthetic examples."""
        all_synthetic = []

        # Generate to fill gaps
        if 'contact_sender' in gaps:
            count = gaps['contact_sender']
            print(f"\nGenerating {count} contact_sender examples...")
            all_synthetic.extend(self.generate_contact_sender_examples(count))

        if 'follow_up' in gaps:
            count = gaps['follow_up']
            print(f"Generating {count} follow_up examples...")
            all_synthetic.extend(self.generate_follow_up_examples(count))

        if 'create_reminder' in gaps:
            count = gaps['create_reminder']
            print(f"Generating {count} create_reminder examples...")
            all_synthetic.extend(self.generate_diverse_reminders(count))

        # Also add boundary cases for NONE
        boundary_count = 50
        print(f"Generating {boundary_count} NONE boundary cases...")
        all_synthetic.extend(self.generate_none_boundary_cases(boundary_count))

        print(f"\nTotal synthetic examples generated: {len(all_synthetic)}")
        return all_synthetic


def main():
    print("=" * 70)
    print("Synthetic Example Generation for Action Dataset")
    print("=" * 70)

    # Load existing annotated data
    print("\nLoading annotated data...")
    with open('data/generation/annotated_actions_initial.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    print(f"Loaded {len(existing_data)} existing annotations")

    # Generate synthetic examples
    generator = SyntheticExampleGenerator(existing_data)
    gaps = generator.analyze_gaps()

    if gaps:
        synthetic_examples = generator.generate_all_synthetic(gaps)

        # Combine with existing
        combined_data = existing_data + synthetic_examples

        # Save combined dataset
        output_file = 'data/generation/annotated_actions_with_synthetic.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved combined dataset to: {output_file}")
        print(f"Total records: {len(combined_data)}")
        print(f"  Real: {len(existing_data)}")
        print(f"  Synthetic: {len(synthetic_examples)}")

        # Final distribution
        action_counts = {}
        for record in combined_data:
            action_type = record['action_type']
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        print("\nFinal action distribution:")
        for action_type, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(combined_data) * 100
            print(f"  {action_type:25s}: {count:5d} ({pct:5.1f}%)")

        return combined_data
    else:
        print("\nNo gaps identified. Using existing dataset.")
        return existing_data


if __name__ == "__main__":
    main()
