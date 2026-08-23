"""Noisy Targeted Synthetic Gap Generator.

Generates synthetic training records targeting categories rare in corporate email corpora
(e.g., SECURITY alerts, 2FA codes, TRANSACTIONAL receipts, shipping confirmations).
Injects realistic noise: typos, quoted chains, signatures, length variance, and conflicting urgency signals.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from ml.schema import CanonicalEmailExample, format_namespaced_id

GAP_TEMPLATES: List[Dict[str, Any]] = [
    {
        "intent": "security",
        "priority": "high",
        "subject": "Security Alert: New Login from Unknown IP ({ip})",
        "body": "We detected a login attempt to your account from an unrecognized location ({city}). If this was you, ignore this message. If not, reset your password immediately.",
        "reasons": ["security", "action_required"],
    },
    {
        "intent": "security",
        "priority": "high",
        "subject": "Your 2FA Verification Code: {code}",
        "body": "Use verification code {code} to complete your sign-in. This code expires in 10 minutes. Do not share this code with anyone.",
        "reasons": ["security", "time_sensitive"],
    },
    {
        "intent": "transactional",
        "priority": "low",
        "subject": "Order Confirmation #{order_id}",
        "body": "Thank you for your purchase from {store}! Your order #{order_id} has been received and is being processed. Total: ${amount}.",
        "reasons": ["transactional"],
    },
    {
        "intent": "transactional",
        "priority": "low",
        "subject": "Your Package Has Shipped - Track #{tracking_id}",
        "body": "Great news! Item #{order_id} is on its way via {carrier}. Track your shipment status using tracking number {tracking_id}.",
        "reasons": ["transactional"],
    },
    {
        "intent": "notification",
        "priority": "low",
        "subject": "Scheduled Maintenance Notice: {date}",
        "body": "System maintenance is scheduled for {date} at midnight UTC. Services may be unavailable for up to 30 minutes. Thank you for your patience.",
        "reasons": ["notification"],
    },
    {
        "intent": "meeting",
        "priority": "medium",
        "subject": "Reschedule: Project Sync with {name}",
        "body": "Could we move our sync call from Thursday to Friday at 2 PM? Please let me know if that time works for you.",
        "reasons": ["meeting", "request"],
    },
    {
        "intent": "request",
        "priority": "high",
        "subject": "Production Database Down - Urgent Fix Needed",
        "body": "The primary database connection pool is throwing connection refused errors. Production environment is down.",
        "reasons": ["system_outage", "action_required"],
    },
    {
        "intent": "promotion",
        "priority": "low",
        "subject": "Urgent Discount Reading: 30% Off Weekend Special!",
        "body": "Don't miss out on our urgent weekend promotion! Use coupon code SAVE30 to get 30% off your next renewal.",
        "reasons": ["promotional"],
    },
]

NAMES = ["Alice Smith", "Robert Chen", "James Wilson", "Sofia Patel", "Elena Rostova"]
STORES = ["TechMart", "CloudStore", "DevGear", "BookDepot"]
CARRIERS = ["FedEx", "UPS", "DHL"]
CITIES = ["Frankfurt, DE", "Tokyo, JP", "Sydney, AU", "London, UK"]


def inject_noise(text: str, rng: random.Random) -> str:
    """Inject controlled noise: typos, quoted reply chains, and signature blocks."""
    res = text

    # Typo injection (15% chance)
    if rng.random() < 0.15:
        typo_map = {"please": "pls", "thanks": "thx", "urgent": "urgnt", "confirm": "confrim"}
        for k, v in typo_map.items():
            if k in res.lower():
                res = re.sub(k, v, res, flags=re.IGNORECASE)

    # Signature block (40% chance)
    if rng.random() < 0.40:
        sig_name = rng.choice(NAMES)
        res += f"\n\nThanks,\n{sig_name}\nSenior Operations Team"

    # Quoted reply chain (20% chance)
    if rng.random() < 0.20:
        prev_sender = rng.choice(NAMES)
        res += f"\n\n> On Previous Date, {prev_sender} wrote:\n> Original message content regarding previous thread."

    # Forward header (15% chance)
    if rng.random() < 0.15:
        fwd_sender = rng.choice(NAMES)
        res = f"---------- Forwarded message ----------\nFrom: {fwd_sender}\n\n" + res

    return res


def generate_synthetic_examples(
    count: int = 100,
    seed: int = 42,
) -> List[CanonicalEmailExample]:
    """Generate specified count of targeted synthetic email examples."""
    rng = random.Random(seed)
    examples: List[CanonicalEmailExample] = []

    for idx in range(1, count + 1):
        tmpl = rng.choice(GAP_TEMPLATES)

        # Format template placeholders
        fmt_kwargs = {
            "ip": f"{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}.{rng.randint(1,255)}",
            "city": rng.choice(CITIES),
            "code": str(rng.randint(100000, 999999)),
            "order_id": str(rng.randint(10000, 99999)),
            "tracking_id": f"TRK{rng.randint(1000000, 9999999)}",
            "amount": f"{rng.randint(10, 500)}.{rng.randint(10, 99):02d}",
            "store": rng.choice(STORES),
            "carrier": rng.choice(CARRIERS),
            "date": "2026-09-15",
            "name": rng.choice(NAMES),
        }

        sbj = tmpl["subject"].format(**fmt_kwargs)
        bdy = tmpl["body"].format(**fmt_kwargs)
        bdy_noisy = inject_noise(bdy, rng)

        raw_id = f"synthetic_{idx:06d}"
        namespaced_id = format_namespaced_id("synthetic", raw_id)

        ex = CanonicalEmailExample(
            id=namespaced_id,
            subject=sbj,
            body=bdy_noisy,
            intent=tmpl["intent"],
            priority=tmpl["priority"],
            priority_reasons=tmpl["reasons"],
            source="synthetic",
            label_source="rules",
            label_confidence=1.0,
            rule_score=4.0,
            language="en",
            source_group_id=f"syn_grp_{tmpl['intent']}",
            is_synthetic=True,
            provenance="synthetic_gap_generator",
        )
        examples.append(ex)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted Noisy Synthetic Gap Generator CLI")
    parser.add_argument("--output", type=str, default="artifacts/synthetic_gaps.jsonl", help="Output path")
    parser.add_argument("--count", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    examples = generate_synthetic_examples(count=args.count, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")

    print(json.dumps({"status": "success", "generated_count": len(examples), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
