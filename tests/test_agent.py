import pandas as pd
import pytest
from click.testing import CliRunner

from hiver_support.agent import IntentModel, ReplyRetriever, SupportAgent, route_message, save_agent
from hiver_support.cli import cli
from hiver_support.evaluation import judge_human_agreement
from hiver_support.labeling import make_golden_template, validate_golden_labels, weak_label


def test_weak_label_is_deterministic():
    assert weak_label("My checked baggage is missing") == "baggage_issue"
    assert weak_label("How do I check in online?") == "check_in_or_boarding"


def test_transactional_intent_escalates_even_when_confident():
    route = route_message("I need a refund", "refund_or_payment", 0.99, 0.99)
    assert route.decision == "escalate"
    assert "trip-specific" in route.reason


def test_safe_high_confidence_request_can_auto_handle():
    route = route_message("How do I check in?", "check_in_or_boarding", 0.92, 0.42)
    assert route.decision == "auto_handle"


def test_case_specific_historical_flight_cannot_auto_handle():
    route = route_message("Why is my flight delayed?", "flight_disruption_or_status", 0.92, 0.42, "DL1076 is delayed")
    assert route.decision == "escalate"


def test_golden_template_is_blinded_and_validation_requires_labels():
    source = pd.DataFrame({
        "customer_tweet_id": ["1", "2"],
        "customer_message": ["My luggage is missing", "How do I check in?"],
    })
    template = make_golden_template(source, total=150)
    assert "silver_intent" not in template
    assert validate_golden_labels(template)


def test_judge_agreement_uses_only_paired_scores():
    scores = pd.DataFrame({
        "human_overall_score": [5, 3, None],
        "llm_overall_score": [5, 2, 4],
    })
    report = judge_human_agreement(scores)
    assert report["n"] == 2
    assert report["exact_agreement"] == 0.5


def test_judge_agreement_rejects_missing_pairs():
    with pytest.raises(ValueError, match="No completed"):
        judge_human_agreement(pd.DataFrame({"human_overall_score": [], "llm_overall_score": []}))


def test_review_queue_keeps_gold_labels_blank(tmp_path):
    history = pd.DataFrame({
        "customer_tweet_id": ["1", "2", "3", "4"],
        "customer_message": ["How do I check in?", "Check in for my flight", "My bag is missing", "Missing checked bag"],
        "historical_reply": ["Try check-in online.", "Try check-in online.", "Please contact baggage support.", "Please contact baggage support."],
        "silver_intent": ["check_in_or_boarding", "check_in_or_boarding", "baggage_issue", "baggage_issue"],
    })
    artifact = tmp_path / "agent.joblib"
    save_agent(SupportAgent(IntentModel().fit(history.customer_message, history.silver_intent), ReplyRetriever().fit(history)), artifact)
    template = tmp_path / "template.csv"
    pd.DataFrame({"customer_tweet_id": ["3"], "customer_message": ["How do I check in?"], "gold_intent": [""], "gold_route": [""]}).to_csv(template, index=False)
    output = tmp_path / "review.csv"
    result = CliRunner().invoke(cli, ["prepare-review-queue", "--input", str(template), "--artifact", str(artifact), "--output", str(output)])
    assert result.exit_code == 0, result.output
    queue = pd.read_csv(output, dtype=str).fillna("")
    assert queue.loc[0, "review_status"] == "needs_human_review"
    assert queue.loc[0, "gold_intent"] == ""
    assert queue.loc[0, "gold_route"] == ""


def test_independent_review_export_hides_suggestions(tmp_path):
    source = tmp_path / "queue.csv"
    pd.DataFrame({
        "customer_tweet_id": ["1", "2"],
        "customer_message": ["Can I get a refund?", "How do I check in?"],
        "suggested_intent": ["refund_or_payment", "check_in_or_boarding"],
        "suggested_route": ["escalate", "auto_handle"],
    }).to_csv(source, index=False)
    output = tmp_path / "independent.csv"
    result = CliRunner().invoke(cli, ["sample-independent-review", "--input", str(source), "--output", str(output), "--n", "30"])
    assert result.exit_code == 0, result.output
    reviewed = pd.read_csv(output, dtype=str).fillna("")
    assert "suggested_intent" not in reviewed
    assert set(reviewed.columns) == {"customer_tweet_id", "customer_message", "reviewed_intent", "reviewed_route", "review_notes"}


def test_promote_reviewed_gold_uses_reviewed_not_suggested_columns(tmp_path):
    source = tmp_path / "queue.csv"
    pd.DataFrame({
        "customer_tweet_id": [str(i) for i in range(150)],
        "customer_message": ["How do I check in?"] * 150,
        "suggested_intent": ["other_or_ambiguous"] * 150,
        "suggested_route": ["escalate"] * 150,
        "reviewed_intent": ["check_in_or_boarding"] * 150,
        "reviewed_route": ["auto_handle"] * 150,
        "review_notes": ["Reviewed independently"] * 150,
    }).to_csv(source, index=False)
    output = tmp_path / "golden.csv"
    result = CliRunner().invoke(cli, ["promote-reviewed-gold", "--input", str(source), "--output", str(output)])
    assert result.exit_code == 0, result.output
    golden = pd.read_csv(output, dtype=str).fillna("")
    assert golden.loc[0, "gold_intent"] == "check_in_or_boarding"
    assert golden.loc[0, "gold_route"] == "auto_handle"
    assert golden.loc[0, "annotator_notes"] == "Reviewed independently"
