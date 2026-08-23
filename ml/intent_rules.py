"""Lightweight intent rule engine for weak labeling.

Evaluates email text and returns heuristic weak intent label, score, and reasons.
Supports canonical taxonomy: SECURITY, TRANSACTIONAL, MEETING, REQUEST, QUESTION,
NOTIFICATION, PROMOTION, COMPLAINT, FOLLOW_UP, INFORMATION, OTHER.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple


INTENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "security": {
        "keywords": ["password reset", "2fa", "verification code", "suspicious login", "security alert", "authentication", "unauthorized"],
        "subject_patterns": [r"security alert", r"password reset", r"verify your account", r"unusual activity"],
        "weight": 5.0,
    },
    "transactional": {
        "keywords": ["order confirmation", "receipt", "invoice", "payment received", "tracking number", "shipped", "package delivered", "billing"],
        "subject_patterns": [r"order #?\d+", r"receipt for", r"your order", r"payment confirmation", r"shipping update"],
        "weight": 4.5,
    },
    "meeting": {
        "keywords": ["meeting", "calendar invite", "reschedule", "zoom link", "teams call", "appointment", "schedule a call", "sync"],
        "subject_patterns": [r"meeting", r"call:", r"calendar", r"invitation:", r"reschedule"],
        "weight": 4.0,
    },
    "request": {
        "keywords": ["please review", "could you send", "can you provide", "action required", "need your input", "please confirm", "kindly send"],
        "subject_patterns": [r"action required", r"request for", r"please review", r"kindly"],
        "weight": 3.5,
    },
    "question": {
        "keywords": ["what is", "how do I", "could you clarify", "when will", "where can", "do you know", "is it possible to"],
        "subject_patterns": [r"\?", r"question regarding", r"inquiry about", r"how to"],
        "weight": 3.5,
    },
    "promotion": {
        "keywords": ["discount", "special offer", "limited time", "% off", "sale ends", "coupon", "promo code", "subscribe now"],
        "subject_patterns": [r"\d+% off", r"sale", r"special offer", r"exclusive deal"],
        "weight": 3.0,
    },
    "notification": {
        "keywords": ["system update", "scheduled maintenance", "policy update", "digest", "newsletter", "terms of service"],
        "subject_patterns": [r"notification:", r"update:", r"maintenance", r"digest"],
        "weight": 2.5,
    },
    "complaint": {
        "keywords": ["issue with", "unacceptable", "disappointed", "broken", "terrible service", "demand refund", "not working"],
        "subject_patterns": [r"complaint", r"issue with", r"disappointed", r"urgent problem"],
        "weight": 3.5,
    },
    "follow_up": {
        "keywords": ["following up on", "checking in on", "reminder:", "any updates on", "previous email", "status update on"],
        "subject_patterns": [r"following up", r"follow up", r"checking in", r"reminder:"],
        "weight": 3.5,
    },
    "information": {
        "keywords": ["fyi", "for your information", "here is the report", "attachment included", "summary of", "notes from"],
        "subject_patterns": [r"fyi:?", r"report:", r"summary:", r"notes:"],
        "weight": 2.0,
    },
}


class IntentRuleEngine:
    """Lightweight pattern matcher for weak intent labeling."""

    def __init__(self) -> None:
        self.patterns = INTENT_PATTERNS
        self.default_intent = "other"

    def predict_weak_intent(self, subject: str, body: str) -> Tuple[str, float, List[str]]:
        """Predict weak intent label, heuristic score, and reasoning list."""
        sbj_lower = (subject or "").lower()
        bdy_lower = (body or "").lower()
        full_text = f"{sbj_lower}\n{bdy_lower}".strip()

        scores: Dict[str, float] = {}
        reasonings: Dict[str, List[str]] = {}

        for intent, config in self.patterns.items():
            score = 0.0
            reasons = []

            # Subject pattern match
            for sp in config["subject_patterns"]:
                if re.search(sp, sbj_lower):
                    score += 2.0
                    reasons.append(f"Subject pattern '{sp}' matched")

            # Keyword match
            kw_count = sum(1 for kw in config["keywords"] if kw in full_text)
            if kw_count > 0:
                score += kw_count * 1.0
                reasons.append(f"{kw_count} keyword(s) matched")

            score *= config["weight"]
            scores[intent] = score
            reasonings[intent] = reasons

        max_score = max(scores.values()) if scores else 0.0
        if max_score >= 2.0:
            best_intent = max(scores, key=scores.get)
            return best_intent, scores[best_intent], reasonings[best_intent]

        return self.default_intent, 0.5, ["No strong intent pattern matched"]
