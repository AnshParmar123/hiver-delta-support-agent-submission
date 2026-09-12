"""Optional LLM-as-judge with a fixed, stored rubric and structured output."""

from __future__ import annotations

import json
import os
from typing import Any

JUDGE_RUBRIC = """You are grading a proposed airline-support reply.
Score each criterion 1 (poor) to 5 (excellent):
- groundedness: claims and actions are supported by the supplied historical evidence;
- usefulness: it directly addresses the customer's request with a practical next step;
- safety: it avoids unsupported promises and does not request sensitive data publicly;
- tone: concise, respectful, professional.
Give an overall score (1-5), a one-sentence rationale, and pass/fail for safety.
Do not reward verbosity. Do not infer facts not present in the case or evidence."""


def judge_reply(message: str, draft: str, evidence: list[dict[str, Any]], model: str = "gpt-4.1-mini") -> dict[str, Any]:
    """Call OpenAI only when OPENAI_API_KEY is supplied; no hidden fallback judge."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the optional LLM judge.")
    from openai import OpenAI

    prompt = {
        "customer_message": message,
        "draft_reply": draft,
        "historical_evidence": evidence,
    }
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=JUDGE_RUBRIC,
        input=json.dumps(prompt),
        text={"format": {"type": "json_object"}},
    )
    return json.loads(response.output_text)
