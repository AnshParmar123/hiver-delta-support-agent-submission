"""Local human-review UI for the Delta support-agent assignment."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hiver_support.agent import SupportAgent, load_agent

ARTIFACT = ROOT / "artifacts" / "delta_agent.joblib"


@st.cache_resource
def get_agent() -> SupportAgent:
    return load_agent(ARTIFACT)


EXAMPLES = {
    "Carry-on policy": "Can I bring a desktop PC tower as a carry-on if it fits the carry-on dimensions?",
    "Flight delay": "My flight has been delayed for two hours. What should I do?",
    "Missing baggage": "My checked bag has not arrived and I need help finding it.",
    "Booking change": "Can I change my flight tomorrow without paying a fee?",
}

st.set_page_config(page_title="Delta Support Review", page_icon="D", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 2.5rem; }
      h1 { font-size: 2rem !important; }
      [data-testid="stMetric"] { border-left: 3px solid #1f77b4; padding-left: 0.8rem; }
      [data-testid="stSidebar"] { border-right: 1px solid #dfe3e8; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Review queue")
    selected_example = st.radio("Load example", list(EXAMPLES), index=0)
    if st.button("Use selected example", use_container_width=True):
        st.session_state.customer_message = EXAMPLES[selected_example]
    st.divider()
    st.caption("Auto-handle: only high-confidence, non-transactional requests with safe historical evidence.")

st.title("Delta Support Review")
st.caption("Evidence-backed customer-support drafts for operator review")

if not ARTIFACT.exists():
    st.error("Model artifact is missing. Run: `hiver-support train --exclude-ids data/evaluation/golden_template.csv`")
    st.stop()

with st.form("review_form", border=False):
    message = st.text_area(
        "Customer message",
        key="customer_message",
        placeholder="Example: My flight has been delayed for two hours. What should I do?",
        height=130,
    )
    submitted = st.form_submit_button("Analyze message", type="primary")

if submitted:
    if not message.strip():
        st.error("Enter a customer message before analysis.")
        st.stop()
    result = get_agent().respond(message.strip())
    route_is_auto = result["decision"] == "auto_handle"

    intent, confidence, route = st.columns(3)
    intent.metric("Predicted intent", result["predicted_intent"].replace("_", " ").title())
    confidence.metric("Intent confidence", f"{result['intent_confidence']:.0%}")
    route.metric("Route", "Auto-handle" if route_is_auto else "Escalate")

    if route_is_auto:
        st.success(result["decision_reason"])
    else:
        st.warning(result["decision_reason"])

    st.subheader("Proposed reply")
    if route_is_auto:
        st.write(result["draft_reply"])
    else:
        st.info("Escalated: this draft is reference material for the human agent and cannot be auto-sent.")
        st.write(result["draft_reply"])

    st.subheader("Historical evidence")
    for index, evidence in enumerate(result["evidence"], start=1):
        with st.expander(f"Similar case {index} | similarity {evidence['similarity']:.0%}", expanded=index == 1):
            st.caption(f"Source customer tweet: {evidence['source_customer_tweet_id']}")
            st.markdown("**Historical customer message**")
            st.write(evidence["similar_customer_message"])
            st.markdown("**Historical Delta response**")
            st.write(evidence["historical_reply"])

st.divider()
st.caption("Local-only review interface. Model: TF-IDF + logistic regression, TF-IDF retrieval, explicit safety-routing rules.")
