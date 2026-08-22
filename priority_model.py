import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


class Prioritizer:
    """Lightweight feedback-adaptive priority model.

    The Q-table is used as an online adjustment to a transparent baseline score.
    This component is intentionally retained as a baseline/personalization layer;
    LLM semantic analysis is implemented separately in the ``llm`` package.
    """

    def __init__(self, q_table_file="q_table.json", reward_history_file="reward_history.json"):
        self.q_table_file = q_table_file
        self.reward_history_file = reward_history_file
        self.q_table = self._load_json(q_table_file, {})
        self.reward_history = self._load_json(reward_history_file, [])
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

    @staticmethod
    def _load_json(path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return default

    def _save_json(self, path: str, value) -> None:
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2)
        except OSError:
            pass

    def _extract_features(self, email: Dict) -> str:
        features = [f"tag_{email.get('tag', 'GENERAL')}"]
        confidence = float(email.get("tag_confidence", 0.0))
        features.append("high_confidence" if confidence > 0.7 else "medium_confidence" if confidence > 0.4 else "low_confidence")
        sentiment = float(email.get("sentiment_score", 0.0))
        features.append("positive_sentiment" if sentiment > 0.1 else "negative_sentiment" if sentiment < -0.1 else "neutral_sentiment")
        metrics = email.get("metrics", {})
        features.append(f"urgency_{metrics.get('urgency', 'low')}")
        features.append(f"intent_{metrics.get('intent', 'general')}")
        if metrics.get("has_deadline", False):
            features.append("has_deadline")
        return "_".join(sorted(features))

    def _calculate_base_score(self, email: Dict) -> float:
        tag_scores = {
            "URGENT": 10.0, "SECURITY": 9.0, "MEETING": 8.0,
            "FINANCIAL": 7.0, "IMPORTANT": 6.0, "GENERAL": 3.0,
            "PROMOTIONAL": 2.0, "NEWSLETTER": 1.0,
        }
        score = tag_scores.get(email.get("tag", "GENERAL"), 3.0)
        score += float(email.get("tag_confidence", 0.0)) * 2.0
        metrics = email.get("metrics", {})
        score += {"high": 3.0, "medium": 1.5, "low": 0.0}.get(metrics.get("urgency", "low"), 0.0)
        score += 2.0 if metrics.get("has_deadline", False) else 0.0
        score += {"request": 2.0, "question": 1.5, "complaint": 2.5, "urgent": 3.0, "meeting": 2.0}.get(metrics.get("intent", "general"), 0.0)
        return max(score, 0.1)

    def predict_priority(self, message_text: str) -> float:
        """Return a normalized baseline priority score for legacy callers."""
        text = (message_text or "").lower()
        score = 0.25
        if any(word in text for word in ("urgent", "asap", "immediately", "critical")):
            score += 0.55
        elif any(word in text for word in ("deadline", "meeting", "please")):
            score += 0.25
        return min(1.0, score)

    def prioritize_emails(self, emails: List[Dict]) -> List[Tuple[float, Dict]]:
        scored = []
        for email in emails:
            state = self._extract_features(email)
            score = self._calculate_base_score(email) + float(self.q_table.get(state, 0.0))
            scored.append((score, email))
        return sorted(scored, key=lambda item: item[0], reverse=True)

    def update(self, email: Dict, user_feedback: float) -> None:
        state = self._extract_features(email)
        current_q = float(self.q_table.get(state, 0.0))
        reward = max(-1.0, min(1.0, float(user_feedback)))
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[state] = new_q
        self.reward_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "reward": reward,
            "old_q": current_q,
            "new_q": new_q,
        })
        self._save_json(self.q_table_file, self.q_table)
        self._save_json(self.reward_history_file, self.reward_history)

    def get_learning_stats(self) -> Dict:
        rewards = [float(item.get("reward", 0.0)) for item in self.reward_history[-100:]]
        return {
            "total_states": len(self.q_table),
            "total_feedback": len(self.reward_history),
            "total_episodes": len(self.reward_history),
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        }

    def get_top_learned_patterns(self, limit: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.q_table.items(), key=lambda item: item[1], reverse=True)[:limit]

    def reset_learning(self) -> None:
        self.q_table = {}
        self.reward_history = []
        self._save_json(self.q_table_file, self.q_table)
        self._save_json(self.reward_history_file, self.reward_history)


# Backward-compatible name used by the original application entry point.
PriorityModel = Prioritizer
