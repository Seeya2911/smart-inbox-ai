"""Deterministic SmartBrief v3 compatibility layer.

This module is retained as the rule-based baseline for comparison with the new
LLM analysis stack. It deliberately contains no LLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SmartSummarizerV3:
    """Context-aware deterministic baseline summarizer.

    The LLM modernization uses this component as a reproducible baseline and
    fallback. Public methods are kept compatible with the historical project.
    """

    PLATFORM_LIMITS = {
        "whatsapp": 50,
        "email": 100,
        "slack": 60,
        "teams": 80,
        "instagram": 40,
        "discord": 60,
    }

    def __init__(
        self,
        context_file: str = "message_context.json",
        max_context_messages: int = 3,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.context_file = context_file
        self.max_context_messages = max_context_messages
        self.confidence_threshold = confidence_threshold
        self.context_data = self._load_context()
        self.stats: Dict[str, Any] = {
            "processed": 0,
            "context_used": 0,
            "platforms": {},
            "intents": {},
            "urgency_levels": {},
            "unique_users": set(),
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        """Parse timestamps into timezone-aware UTC datetimes."""
        if not value:
            return datetime.now(timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _load_context(self) -> Dict[str, Any]:
        if not os.path.exists(self.context_file):
            return {"conversations": {}, "user_profiles": {}}
        try:
            with open(self.context_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            data.setdefault("conversations", {})
            data.setdefault("user_profiles", {})
            # Do not silently discard historical context during construction.
            # Retention is enforced when new data is stored, which keeps old
            # exported research fixtures readable and deterministic.
            return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load context %s: %s", self.context_file, exc)
            return {"conversations": {}, "user_profiles": {}}

    def _save_context(self) -> None:
        try:
            parent = os.path.dirname(os.path.abspath(self.context_file))
            os.makedirs(parent, exist_ok=True)
            with open(self.context_file, "w", encoding="utf-8") as handle:
                json.dump(self.context_data, handle, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Could not save context: %s", exc)

    def _cleanup_old_context(self, data: Dict[str, Any]) -> None:
        """Remove messages older than 30 days when explicitly requested."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        for key, messages in data.get("conversations", {}).items():
            data["conversations"][key] = [
                message
                for message in messages
                if self._parse_datetime(message.get("timestamp")) > cutoff
            ]

    def _get_context_key(self, user_id: str, platform: str) -> str:
        return f"{user_id}_{platform}"

    def _extract_context(self, user_id: str, platform: str) -> List[Dict[str, Any]]:
        messages = self.context_data.get("conversations", {}).get(
            self._get_context_key(user_id, platform), []
        )
        return messages[-self.max_context_messages :]

    def get_user_context(self, user_id: str, platform: str) -> List[Dict[str, Any]]:
        return self._extract_context(user_id, platform)

    def _store_message_context(self, message_data: Dict[str, Any]) -> None:
        key = self._get_context_key(
            str(message_data.get("user_id", "unknown")),
            str(message_data.get("platform", "unknown")),
        )
        conversations = self.context_data.setdefault("conversations", {})
        history = conversations.setdefault(key, [])
        history.append(
            {
                "message_text": str(message_data.get("message_text", "")),
                "timestamp": message_data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "message_id": message_data.get("message_id") or f"msg_{datetime.now().timestamp()}",
            }
        )
        conversations[key] = history[-self.max_context_messages * 2 :]
        self._save_context()

    @staticmethod
    def _word_count(text: str, pattern: str) -> int:
        return len(re.findall(pattern, text, flags=re.IGNORECASE))

    def _classify_intent(
        self, text: str, context_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, float]:
        normalized = text.strip().lower()

        # Explicit request constructions take precedence over the generic '?'
        # signal. This fixes "Can you send me ...?" being classified as a question.
        request_patterns = [
            r"\bcan you\s+(send|share|provide|review|check|help|prepare|forward|update)\b",
            r"\bcould you\s+(send|share|provide|review|check|help|prepare|forward|update)\b",
            r"\bwould you\s+(send|share|provide|review|check|help|prepare|forward|update)\b",
            r"\bplease\s+\w+",
            r"\bsend me\b",
            r"\bi need\b",
            r"\bneed (?:you|a|the|this|that)\b",
        ]
        if any(re.search(pattern, normalized) for pattern in request_patterns):
            return "request", 0.9

        if context_messages and any(
            token in normalized for token in ("update", "status", "progress", "any news", "follow up", "follow-up")
        ):
            current_words = set(re.findall(r"\b\w+\b", normalized))
            for previous in context_messages:
                previous_words = set(
                    re.findall(r"\b\w+\b", str(previous.get("message_text", "")).lower())
                )
                if len(current_words & previous_words) >= 2:
                    return "follow_up", 0.9

        rules = [
            ("complaint", [r"\bnot working\b", r"\bproblem\b", r"\bissue\b", r"\berror\b", r"\bbug\b", r"\bbroken\b"]),
            ("appreciation", [r"\bthank(?:s| you)?\b", r"\bappreciate\b", r"\bwell done\b", r"\bgreat job\b"]),
            ("urgent", [r"\burgent\b", r"\basap\b", r"\bemergency\b", r"\bcritical\b", r"\bimmediately\b"]),
            ("social", [r"\bhow are you\b", r"\bwhat's up\b", r"\bhang out\b", r"\bparty\b"]),
            ("informational", [r"\bfyi\b", r"\bfor your information\b", r"\bjust letting you know\b", r"\bheads up\b"]),
            ("confirmation", [r"\bconfirmed\b", r"\bgot it\b", r"\bunderstood\b", r"\bsounds good\b"]),
            ("schedule", [r"\bmeeting\b", r"\bappointment\b", r"\bschedule\b", r"\bcalendar\b"]),
            ("follow_up", [r"\bfollow[- ]?up\b", r"\bany update\b", r"\bany news\b", r"\bheard back\b"]),
            ("question", [r"\bwhat\b", r"\bhow\b", r"\bwhen\b", r"\bwhere\b", r"\bwhy\b", r"\bwhich\b", r"\bwho\b", r"\?"]),
        ]
        for intent, patterns in rules:
            score = sum(self._word_count(normalized, pattern) for pattern in patterns)
            if score:
                return intent, min(1.0, 0.65 + score * 0.1)
        return "informational", 0.3

    def _analyze_urgency(
        self, text: str, context_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, float]:
        normalized = text.lower()
        high = ["urgent", "asap", "emergency", "critical", "immediately", "right now", "deadline today", "today", "in 2 hours", "in two hours", "within 2 hours", "within two hours"]
        medium = ["soon", "quickly", "priority", "important", "deadline", "by tomorrow", "this week", "tomorrow"]
        low = ["when you can", "no rush", "whenever", "no hurry", "take your time"]

        if any(token in normalized for token in high):
            return "high", 0.95
        if any(token in normalized for token in medium):
            return "medium", 0.75
        if any(token in normalized for token in low):
            return "low", 0.9

        if context_messages:
            previous = " ".join(str(m.get("message_text", "")).lower() for m in context_messages)
            current_pressure = sum(token in normalized for token in ("need", "deadline", "tomorrow", "hours", "urgent"))
            previous_pressure = sum(token in previous for token in ("need", "deadline", "tomorrow", "hours", "urgent"))
            if current_pressure > previous_pressure:
                return "high", 0.8

        if len(text) < 50:
            return "low", 0.4
        if "?" in text or any(word in normalized for word in ("need", "want", "require")):
            return "medium", 0.5
        return "low", 0.3

    @staticmethod
    def _make_summary(text: str, limit: int) -> str:
        clean = re.sub(r"\s+", " ", text.strip())
        if len(clean) <= limit:
            return clean
        truncated = clean[: max(0, limit - 3)].rsplit(" ", 1)[0]
        return (truncated or clean[: max(0, limit - 3)]) + "..."

    def summarize(self, message: Dict[str, Any], use_context: bool = True) -> Dict[str, Any]:
        text = str(message.get("message_text", "")).strip()
        user_id = str(message.get("user_id", "unknown"))
        platform = str(message.get("platform", "email")).lower()
        context = self._extract_context(user_id, platform) if use_context else []
        context_used = bool(context)

        intent, intent_confidence = self._classify_intent(text, context)
        urgency, urgency_confidence = self._analyze_urgency(text, context)
        confidence = float(min(intent_confidence, urgency_confidence))

        result = {
            "summary": self._make_summary(text, self.PLATFORM_LIMITS.get(platform, 100)),
            "intent": intent,
            "urgency": urgency,
            "confidence": confidence,
            "intent_confidence": intent_confidence,
            "urgency_confidence": urgency_confidence,
            "context_used": context_used,
            "platform_optimized": True,
            "platform": platform,
            "context_insights": [],
            "message_id": message.get("message_id") or f"msg_{datetime.now().timestamp()}",
            "baseline": True,
        }

        if context:
            result["context_insights"] = ["Related conversation context was available."]

        self.stats["processed"] += 1
        self.stats["context_used"] += int(context_used)
        self.stats["platforms"][platform] = self.stats["platforms"].get(platform, 0) + 1
        self.stats["intents"][intent] = self.stats["intents"].get(intent, 0) + 1
        self.stats["urgency_levels"][urgency] = self.stats["urgency_levels"].get(urgency, 0) + 1
        self.stats["unique_users"].add(user_id)

        if use_context:
            self._store_message_context(message)
        return result

    def batch_summarize(self, messages: List[Dict[str, Any]], use_context: bool = True) -> List[Dict[str, Any]]:
        return [self.summarize(message, use_context=use_context) for message in messages]

    def get_stats(self) -> Dict[str, Any]:
        stats = dict(self.stats)
        stats["unique_users"] = len(self.stats["unique_users"])
        return stats


def summarize_message(message: Dict[str, Any], use_context: bool = True, **kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible convenience wrapper."""
    return SmartSummarizerV3().summarize(message, use_context=use_context)
