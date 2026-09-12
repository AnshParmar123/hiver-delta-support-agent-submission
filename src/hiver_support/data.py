"""Dataset extraction and deterministic, inspectable preprocessing."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REDACTION_PATTERNS = (
    (re.compile(r"https?://\S+|www\.\S+", re.I), "[URL]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b"), "[PHONE]"),
    (re.compile(r"@[A-Za-z0-9_]+"), "[HANDLE]"),
)

USE_COLUMNS = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
]


def redact_text(value: object) -> str:
    """Remove common public identifiers before modelling or exporting examples."""
    text = "" if pd.isna(value) else str(value)
    for pattern, replacement in REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).removesuffix(".0")


def extract_linked_conversations(
    raw_path: Path, brand_handle: str, chunk_size: int = 100_000, max_rows: int | None = None
) -> pd.DataFrame:
    """Return customer tweets with the brand's immediate historical response.

    Two passes avoid loading the 500 MB source file into memory.  We only keep
    direct reply edges, avoiding speculative reconstruction of long, noisy threads.
    """
    brand_lower = brand_handle.casefold().lstrip("@")
    replies: dict[str, dict[str, str]] = {}
    reader = pd.read_csv(raw_path, usecols=USE_COLUMNS, chunksize=chunk_size, dtype=str)
    for chunk in reader:
        is_brand = chunk["author_id"].fillna("").str.casefold().eq(brand_lower)
        is_outbound = chunk["inbound"].fillna("").str.casefold().eq("false")
        for row in chunk.loc[is_brand & is_outbound].itertuples(index=False):
            parent_id = _normalise_id(row.in_response_to_tweet_id)
            if parent_id and parent_id not in replies:
                replies[parent_id] = {
                    "brand_tweet_id": _normalise_id(row.tweet_id),
                    "historical_reply": redact_text(row.text),
                }

    records: list[dict[str, str]] = []
    reader = pd.read_csv(raw_path, usecols=USE_COLUMNS, chunksize=chunk_size, dtype=str)
    for chunk in reader:
        inbound = chunk["inbound"].fillna("").str.casefold().eq("true")
        for row in chunk.loc[inbound].itertuples(index=False):
            tweet_id = _normalise_id(row.tweet_id)
            reply = replies.get(tweet_id)
            if not reply:
                continue
            message = redact_text(row.text)
            if not message or not reply["historical_reply"]:
                continue
            records.append({
                "customer_tweet_id": tweet_id,
                "brand_tweet_id": reply["brand_tweet_id"],
                "created_at": row.created_at or "",
                "customer_message": message,
                "historical_reply": reply["historical_reply"],
            })
            if max_rows and len(records) >= max_rows:
                return pd.DataFrame(records)
    return pd.DataFrame(records)
