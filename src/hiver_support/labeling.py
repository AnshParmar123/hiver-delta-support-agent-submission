"""High-precision silver labels and a human-labelling export."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import INTENTS, SEED

# Ordered from distinctive to broad. These labels train a baseline only; humans label gold.
RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baggage_issue", ("baggage", "luggage", "suitcase", "bag ", "bags ", "lost bag", "checked bag")),
    ("refund_or_payment", ("refund", "charge", "charged", "payment", "credit card", "reimburse", "voucher")),
    ("loyalty_or_account", ("skymiles", "sky miles", "medallion", "mileage", "miles account", "account locked")),
    ("check_in_or_boarding", ("check in", "check-in", "boarding pass", "board the", "boarding")),
    ("flight_disruption_or_status", ("cancelled", "canceled", "delay", "delayed", "late flight", "flight status", "missed connection", "weather")),
    ("booking_change_or_cancellation", ("change my flight", "change flight", "reschedule", "cancel my", "confirmation number", "booking", "reservation")),
    ("general_information", ("what time", "how do i", "where can i", "do you offer", "can i bring", "what is the")),
)


def weak_label(message: str) -> str:
    text = message.casefold()
    for intent, phrases in RULES:
        if any(phrase in text for phrase in phrases):
            return intent
    return "other_or_ambiguous"


def label_silver(frame: pd.DataFrame) -> pd.DataFrame:
    labelled = frame.copy()
    labelled["silver_intent"] = labelled["customer_message"].map(weak_label)
    return labelled


def make_golden_template(frame: pd.DataFrame, total: int = 200) -> pd.DataFrame:
    """Stratified, deterministic sample for human annotation, without reply leakage."""
    labelled = label_silver(frame)
    per_intent = max(1, total // len(INTENTS))
    parts: list[pd.DataFrame] = []
    for intent in INTENTS:
        pool = labelled[labelled["silver_intent"].eq(intent)]
        parts.append(pool.sample(n=min(per_intent, len(pool)), random_state=SEED))
    sample = pd.concat(parts).drop_duplicates("customer_tweet_id")
    remaining = total - len(sample)
    if remaining > 0:
        available = labelled[~labelled["customer_tweet_id"].isin(sample["customer_tweet_id"])]
        sample = pd.concat([sample, available.sample(n=min(remaining, len(available)), random_state=SEED)])
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Do not expose the weak-label stratum to annotators; it would anchor their judgement.
    result = sample[["customer_tweet_id", "customer_message"]].copy()
    result["gold_intent"] = ""
    result["gold_route"] = ""
    result["annotator_notes"] = ""
    return result


def validate_golden_labels(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if len(frame) < 150 or len(frame) > 250:
        issues.append("Golden set must contain 150-250 rows.")
    for column in ("gold_intent", "gold_route"):
        if column not in frame:
            issues.append(f"Missing {column} column.")
        elif frame[column].fillna("").str.strip().eq("").any():
            issues.append(f"{column} contains blank labels.")
    invalid = set(frame.get("gold_intent", pd.Series(dtype=str)).dropna()) - set(INTENTS)
    if invalid:
        issues.append(f"Unknown intent labels: {sorted(invalid)}")
    invalid_routes = set(frame.get("gold_route", pd.Series(dtype=str)).dropna()) - {"auto_handle", "escalate"}
    if invalid_routes:
        issues.append(f"Unknown route labels: {sorted(invalid_routes)}")
    return issues
