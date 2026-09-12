"""Required comparator systems, intentionally simpler than the final agent."""

from __future__ import annotations

import pandas as pd

from .labeling import weak_label


def trivial_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Trivial baseline: never infer a specific intent and never auto-handle."""
    result = frame.copy()
    result["predicted_intent"] = "other_or_ambiguous"
    result["decision"] = "escalate"
    result["decision_reason"] = "Trivial baseline: always defer to a human."
    result["draft_reply"] = ""
    return result


def keyword_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Simple baseline: deterministic keyword rules, with no similarity model."""
    result = frame.copy()
    result["predicted_intent"] = result["customer_message"].map(weak_label)
    safe = result.predicted_intent.isin({"general_information", "check_in_or_boarding"})
    result["decision"] = safe.map({True: "auto_handle", False: "escalate"})
    result["decision_reason"] = "Simple baseline: fixed keyword rule."
    result["draft_reply"] = "Thank you for contacting Delta. A support specialist will review your request."
    return result
