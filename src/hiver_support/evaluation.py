"""Metrics and audit exports. No model score is computed before gold labels exist."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score


def intent_metrics(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "n": len(frame),
        "accuracy": round(float(accuracy_score(frame.gold_intent, frame.predicted_intent)), 4),
        "macro_f1": round(float(f1_score(frame.gold_intent, frame.predicted_intent, average="macro", zero_division=0)), 4),
        "per_intent": classification_report(frame.gold_intent, frame.predicted_intent, output_dict=True, zero_division=0),
    }


def routing_metrics(frame: pd.DataFrame) -> dict[str, object]:
    auto = frame[frame.decision.eq("auto_handle")]
    return {
        "coverage": round(float(len(auto) / len(frame)), 4) if len(frame) else 0.0,
        "auto_route_precision": round(float((auto.gold_route == "auto_handle").mean()), 4) if len(auto) else None,
        "escalation_recall": round(float((frame[frame.gold_route == "escalate"].decision == "escalate").mean()), 4)
        if (frame.gold_route == "escalate").any() else None,
    }


def evaluate_predictions(frame: pd.DataFrame, output_path: Path) -> dict[str, object]:
    result = {"intent": intent_metrics(frame), "routing": routing_metrics(frame)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def judge_human_agreement(frame: pd.DataFrame) -> dict[str, object]:
    """Agreement for a 1-5 human and LLM judge score on the same blinded drafts."""
    valid = frame.dropna(subset=["human_overall_score", "llm_overall_score"]).copy()
    if valid.empty:
        raise ValueError("No completed human/LLM paired scores.")
    exact = (valid.human_overall_score.astype(int) == valid.llm_overall_score.astype(int)).mean()
    return {
        "n": len(valid),
        "exact_agreement": round(float(exact), 4),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(
            valid.human_overall_score.astype(int), valid.llm_overall_score.astype(int), weights="quadratic"
        )), 4),
    }
