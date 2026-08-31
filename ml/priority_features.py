"""Priority feature extraction from raw email text for Smart Inbox AI.

Extracts text-derived urgency, deadline, security, and structural signals
from email subject and body ONLY. Zero metadata leakage.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from scipy.sparse import csr_matrix

from ml.deduplication import strip_email_boilerplate
from ml.schema import CanonicalEmailExample

# Regex patterns for observable urgency and priority indicators in email text
_URGENT_SUBJECT_RE = re.compile(
    r"\b(urgent|urgently|asap|immediately|critical|emergency|action\s+required|time[\s-]sensitive|high\s+priority|immediate\s+attention)\b",
    re.IGNORECASE,
)

_URGENT_BODY_RE = re.compile(
    r"\b(urgent|urgently|asap|immediately|critical|emergency|action\s+required|time[\s-]sensitive|deadline|due\s+date|at\s+your\s+earliest|without\s+delay|needs\s+immediate)\b",
    re.IGNORECASE,
)

_DEADLINE_RE = re.compile(
    r"\b(due\s+by|deadline\s+is|by\s+eod|by\s+end\s+of\s+day|by\s+tomorrow|expires\s+on|before\s+\d{1,2}(:\d{2})?\s*(am|pm)?|within\s+\d+\s+(hours?|days?|minutes?)|due\s+today)\b",
    re.IGNORECASE,
)

_SECURITY_RE = re.compile(
    r"\b(compromised|unauthorized|suspicious\s+activity|security\s+alert|security\s+breach|account\s+locked|suspended\s+account|password\s+reset|verify\s+your\s+identity|2fa\s+code|access\s+denied)\b",
    re.IGNORECASE,
)

_FINANCIAL_RE = re.compile(
    r"\b(payment\s+overdue|past\s+due|final\s+notice|invoice\s+overdue|immediate\s+payment|account\s+suspension\s+warning|balance\s+due|late\s+fee)\b",
    re.IGNORECASE,
)

_ACTION_IMPERATIVE_RE = re.compile(
    r"\b(please\s+confirm|please\s+approve|please\s+respond|action\s+needed|needs\s+approval|requires\s+your\s+attention|approval\s+required)\b",
    re.IGNORECASE,
)


class PriorityFeatureExtractor:
    """Extracts numeric feature vectors from raw email subject + body."""

    FEATURE_NAMES = [
        "subject_has_urgent",
        "body_urgent_count",
        "has_deadline_expr",
        "has_security_alert",
        "has_financial_urgency",
        "has_action_imperative",
        "subject_exclamation_count",
        "subject_question_count",
        "subject_caps_ratio",
        "subject_char_length",
        "body_char_length",
    ]

    def extract_features(self, examples: List[CanonicalEmailExample]) -> np.ndarray:
        """Transform a list of canonical examples into a 2D numpy array of shape (N, D)."""
        rows: List[List[float]] = []
        for ex in examples:
            subject = str(ex.subject or "")
            clean_body = strip_email_boilerplate(str(ex.body or ""))
            full_clean = f"{subject}\n{clean_body}"

            # 1. Subject urgency flag
            subj_urgent = 1.0 if _URGENT_SUBJECT_RE.search(subject) else 0.0

            # 2. Body urgency count
            body_urgent_matches = len(_URGENT_BODY_RE.findall(clean_body))
            body_urgent_count = float(min(body_urgent_matches, 5.0)) / 5.0

            # 3. Deadline expression flag
            has_deadline = 1.0 if _DEADLINE_RE.search(full_clean) else 0.0

            # 4. Security breach / alert flag
            has_security = 1.0 if _SECURITY_RE.search(full_clean) else 0.0

            # 5. Financial / payment overdue flag
            has_financial = 1.0 if _FINANCIAL_RE.search(full_clean) else 0.0

            # 6. Action imperative flag
            has_action = 1.0 if _ACTION_IMPERATIVE_RE.search(full_clean) else 0.0

            # 7. Subject exclamation count
            excl_count = float(min(subject.count("!"), 3)) / 3.0

            # 8. Subject question count
            ques_count = float(min(subject.count("?"), 3)) / 3.0

            # 9. Subject uppercase ratio
            alpha_chars = [c for c in subject if c.isalpha()]
            caps_ratio = (
                float(sum(1 for c in alpha_chars if c.isupper())) / len(alpha_chars)
                if len(alpha_chars) >= 5
                else 0.0
            )

            # 10. Normalized subject length
            subj_len = float(min(len(subject), 150)) / 150.0

            # 11. Normalized body length
            body_len = float(min(len(clean_body), 2000)) / 2000.0

            rows.append([
                subj_urgent,
                body_urgent_count,
                has_deadline,
                has_security,
                has_financial,
                has_action,
                excl_count,
                ques_count,
                caps_ratio,
                subj_len,
                body_len,
            ])

        return np.array(rows, dtype=np.float32)

    def extract_sparse(self, examples: List[CanonicalEmailExample]) -> csr_matrix:
        """Extract as csr_matrix for stacking with TF-IDF."""
        dense = self.extract_features(examples)
        return csr_matrix(dense)
