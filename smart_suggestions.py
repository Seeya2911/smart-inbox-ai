"""Action suggestions derived from structured message analysis."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List


class SmartSuggestionsModule:
    """Convert analysis into safe, reviewable UI suggestions."""

    def __init__(self, suggestions_log: str = "suggestions_log.json"):
        self.suggestions_log = suggestions_log
        self.usage_stats = self.load_usage_stats()
        self.suggestion_templates = {
            "URGENT": ["quick_reply", "set_reminder", "escalate"],
            "MEETING": ["add_calendar", "accept_invite", "request_reschedule"],
            "FINANCIAL": ["review_invoice", "approve_payment", "forward_accounting"],
            "PROMOTIONAL": ["archive", "unsubscribe", "save_deal"],
            "SECURITY": ["verify_sender", "report_phishing", "contact_it"],
            "IMPORTANT": ["detailed_reply", "schedule_followup", "delegate"],
            "NEWSLETTER": ["read_later", "unsubscribe", "archive"],
            "GENERAL": ["quick_reply", "read_later", "archive"],
        }
        self.labels = {
            action: action.replace("_", " ").title()
            for actions in self.suggestion_templates.values()
            for action in actions
        }

    def load_usage_stats(self) -> Dict:
        if os.path.exists(self.suggestions_log):
            try:
                with open(self.suggestions_log, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                pass
        return {"action_counts": {}, "success_rates": {}}

    def save_usage_stats(self) -> None:
        try:
            with open(self.suggestions_log, "w", encoding="utf-8") as handle:
                json.dump(self.usage_stats, handle, indent=2)
        except OSError:
            pass

    def generate_suggestions(self, email: Dict, tag: str, confidence: float) -> List[Dict]:
        tag = str(tag or "GENERAL").upper()
        actions = self.suggestion_templates.get(tag, self.suggestion_templates["GENERAL"])
        subject = email.get("subject", "")
        sender = email.get("sender", "Unknown")
        results = []
        for action in actions:
            results.append({
                "action": action,
                "text": self.labels[action],
                "priority": "high" if tag in {"URGENT", "SECURITY"} else "medium",
                "confidence": max(0.0, min(1.0, float(confidence))),
                "context": f"Based on: {subject[:80]}" if subject else f"From: {sender}",
                "estimated_time": "2-5 min",
                "success_rate": self.usage_stats.get("success_rates", {}).get(action, 0.75),
            })
        return results[:5]

    def execute_suggestion(self, email: Dict, action: str) -> Dict:
        """Record a user-selected action; do not perform destructive operations here."""
        known_actions = {a for actions in self.suggestion_templates.values() for a in actions}
        if action not in known_actions:
            return {"success": False, "action": action, "message": "Unknown suggestion action"}
        counts = self.usage_stats.setdefault("action_counts", {})
        counts[action] = int(counts.get(action, 0)) + 1
        self.save_usage_stats()
        return {
            "success": True,
            "action": action,
            "message": f"Recorded action: {action}",
            "timestamp": datetime.now().isoformat(),
        }

    def get_suggestion_stats(self) -> Dict:
        counts = self.usage_stats.get("action_counts", {})
        return {
            "action_counts": dict(counts),
            "success_rates": dict(self.usage_stats.get("success_rates", {})),
            "total_actions": sum(int(value) for value in counts.values()),
        }
