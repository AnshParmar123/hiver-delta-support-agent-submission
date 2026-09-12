"""Interpretable intent, retrieval, and safety-routing components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .config import SEED

SENSITIVE_MARKERS = ("[phone]", "[email]", "confirmation", "credit card", "passport", "ticket number")
TRANSACTIONAL_INTENTS = {
    "booking_change_or_cancellation", "baggage_issue", "refund_or_payment", "loyalty_or_account",
}


class IntentModel:
    """TF-IDF plus multinomial logistic regression; feature weights are inspectable."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True)
        self.classifier = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=SEED)

    def fit(self, messages: pd.Series, labels: pd.Series) -> "IntentModel":
        self.classifier.fit(self.vectorizer.fit_transform(messages), labels)
        return self

    def predict(self, messages: list[str]) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.classifier.predict_proba(self.vectorizer.transform(messages))
        indexes = probabilities.argmax(axis=1)
        return self.classifier.classes_[indexes], probabilities.max(axis=1)

    def top_features(self, intent: str, count: int = 12) -> list[str]:
        index = list(self.classifier.classes_).index(intent)
        weights = self.classifier.coef_[index]
        names = self.vectorizer.get_feature_names_out()
        return names[np.argsort(weights)[-count:][::-1]].tolist()


class ReplyRetriever:
    """Returns a historical brand reply as a draft, preserving provenance."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50_000, sublinear_tf=True)
        self.matrix = None
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "ReplyRetriever":
        self.history = history.reset_index(drop=True).copy()
        self.matrix = self.vectorizer.fit_transform(self.history["customer_message"])
        return self

    def retrieve(self, message: str, intent: str, k: int = 3) -> list[dict[str, object]]:
        if self.history is None or self.matrix is None:
            raise RuntimeError("ReplyRetriever has not been fitted.")
        candidate_idx = np.flatnonzero(self.history["silver_intent"].eq(intent).to_numpy())
        if len(candidate_idx) == 0:
            candidate_idx = np.arange(len(self.history))
        scores = cosine_similarity(self.vectorizer.transform([message]), self.matrix[candidate_idx]).ravel()
        ranked = candidate_idx[np.argsort(scores)[::-1][:k]]
        return [
            {
                "source_customer_tweet_id": self.history.iloc[i]["customer_tweet_id"],
                "similar_customer_message": self.history.iloc[i]["customer_message"],
                "historical_reply": self.history.iloc[i]["historical_reply"],
                "similarity": round(float(score), 4),
            }
            for i, score in zip(ranked, np.sort(scores)[::-1][:k])
        ]


@dataclass(frozen=True)
class RouteDecision:
    decision: str
    reason: str


def route_message(
    message: str, intent: str, confidence: float, retrieval_similarity: float, retrieved_reply: str = ""
) -> RouteDecision:
    lowered = message.casefold()
    if any(marker in lowered for marker in SENSITIVE_MARKERS):
        return RouteDecision("escalate", "Customer message may require account or payment verification.")
    reply_lower = retrieved_reply.casefold()
    # Direct historical-reply reuse is unsafe when it carries facts unique to its source case.
    if "direct message" in reply_lower or " dm " in f" {reply_lower} ":
        return RouteDecision("escalate", "Historical resolution requires a private support interaction.")
    historical_flights = set(re.findall(r"\bdl\s?\d{1,4}\b", reply_lower))
    message_flights = set(re.findall(r"\bdl\s?\d{1,4}\b", lowered))
    if historical_flights - message_flights:
        return RouteDecision("escalate", "Historical resolution contains flight-specific details not present in this message.")
    if intent in TRANSACTIONAL_INTENTS:
        return RouteDecision("escalate", f"{intent} requires a human to verify trip-specific policy or account data.")
    if confidence < 0.70:
        return RouteDecision("escalate", f"Intent confidence {confidence:.2f} is below the 0.70 auto-handle threshold.")
    if retrieval_similarity < 0.20:
        return RouteDecision("escalate", "No sufficiently similar historical resolution was retrieved.")
    if intent in {"general_information", "check_in_or_boarding", "flight_disruption_or_status"}:
        return RouteDecision("auto_handle", "High-confidence, non-transactional request with a similar historical resolution.")
    return RouteDecision("escalate", "Ambiguous request is outside the auto-handle policy.")


class SupportAgent:
    def __init__(self, intent_model: IntentModel, retriever: ReplyRetriever) -> None:
        self.intent_model = intent_model
        self.retriever = retriever

    def respond(self, message: str) -> dict[str, object]:
        intent, confidence = self.intent_model.predict([message])
        examples = self.retriever.retrieve(message, str(intent[0]))
        route = route_message(
            message, str(intent[0]), float(confidence[0]), float(examples[0]["similarity"]),
            str(examples[0]["historical_reply"]),
        )
        return {
            "predicted_intent": str(intent[0]),
            "intent_confidence": round(float(confidence[0]), 4),
            "decision": route.decision,
            "decision_reason": route.reason,
            # Retrieval is deliberate: the draft remains traceable to a real prior resolution.
            "draft_reply": examples[0]["historical_reply"],
            "evidence": examples,
        }


def save_agent(agent: SupportAgent, artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(agent, artifact_path)


def load_agent(artifact_path: Path) -> SupportAgent:
    return joblib.load(artifact_path)
