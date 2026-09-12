from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from .agent import IntentModel, ReplyRetriever, SupportAgent, load_agent, save_agent
from .baselines import keyword_predictions, trivial_predictions
from .config import ARTIFACTS_DIR, BRAND_HANDLE, EVALUATION_DIR, PROCESSED_DIR, RAW_DATA_PATH
from .data import extract_linked_conversations
from .evaluation import evaluate_predictions, judge_human_agreement
from .judge import judge_reply
from .labeling import label_silver, make_golden_template, validate_golden_labels


@click.group()
def cli() -> None:
    """Build and evaluate the auditable Delta support agent."""


@cli.command()
@click.option("--raw-path", type=click.Path(path_type=Path), default=RAW_DATA_PATH)
@click.option("--output", type=click.Path(path_type=Path), default=PROCESSED_DIR / "delta_linked.csv")
@click.option("--max-rows", type=int, default=30_000, show_default=True)
def extract(raw_path: Path, output: Path, max_rows: int) -> None:
    """Extract direct customer -> Delta response pairs from the Kaggle CSV."""
    frame = extract_linked_conversations(raw_path, BRAND_HANDLE, max_rows=max_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    click.echo(f"Wrote {len(frame)} linked conversations to {output}")


@cli.command("prepare-golden")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), default=PROCESSED_DIR / "delta_linked.csv")
@click.option("--output", type=click.Path(path_type=Path), default=EVALUATION_DIR / "golden_template.csv")
@click.option("--n", type=click.IntRange(150, 250), default=200, show_default=True)
def prepare_golden(input_path: Path, output: Path, n: int) -> None:
    """Create a blinded, stratified CSV for genuine human annotation."""
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    template = make_golden_template(frame, total=n)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
    click.echo(f"Wrote {len(template)} annotation rows to {output}")


@cli.command("validate-golden")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), default=EVALUATION_DIR / "golden.csv")
def validate_golden(input_path: Path) -> None:
    """Fail early if the human-labelled golden file is incomplete or invalid."""
    issues = validate_golden_labels(pd.read_csv(input_path, dtype=str).fillna(""))
    if issues:
        raise click.ClickException(" ".join(issues))
    click.echo("Golden set is complete and structurally valid.")


@cli.command("prepare-review-queue")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), default=EVALUATION_DIR / "golden_template.csv")
@click.option("--artifact", type=click.Path(exists=True, path_type=Path), default=ARTIFACTS_DIR / "delta_agent.joblib")
@click.option("--output", type=click.Path(path_type=Path), default=EVALUATION_DIR / "golden_review_queue.csv")
def prepare_review_queue(input_path: Path, artifact: Path, output: Path) -> None:
    """Create AI suggestions for human review; this is not a golden evaluation set."""
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    if "customer_message" not in frame:
        raise click.ClickException("Input must have a customer_message column.")
    agent = load_agent(artifact)
    suggestions = [agent.respond(message) for message in frame.customer_message]
    result = frame.copy()
    result["suggested_intent"] = [item["predicted_intent"] for item in suggestions]
    result["suggested_route"] = [item["decision"] for item in suggestions]
    result["suggested_route_reason"] = [item["decision_reason"] for item in suggestions]
    result["review_status"] = "needs_human_review"
    # Preserve blank gold columns: suggestions must never masquerade as human labels.
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    click.echo(f"Wrote {len(result)} AI-assisted suggestions to {output}. Review before using as gold.")


@cli.command("sample-independent-review")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), default=EVALUATION_DIR / "golden_review_queue.csv")
@click.option("--output", type=click.Path(path_type=Path), default=EVALUATION_DIR / "independent_review_50.csv")
@click.option("--n", type=click.IntRange(30, 100), default=50, show_default=True)
def sample_independent_review(input_path: Path, output: Path, n: int) -> None:
    """Export the highest-risk rows without model suggestions for independent review."""
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    required = {"customer_tweet_id", "customer_message", "suggested_intent", "suggested_route"}
    missing = required - set(frame.columns)
    if missing:
        raise click.ClickException(f"Missing columns: {sorted(missing)}")
    # Prioritize uncertain intent, auto-handle suggestions, and sensitive/transactional language.
    risk = pd.Series(0, index=frame.index, dtype=int)
    risk += frame["suggested_route"].eq("auto_handle").astype(int) * 4
    risk += frame["suggested_intent"].isin({"refund_or_payment", "booking_change_or_cancellation", "baggage_issue", "loyalty_or_account"}).astype(int) * 3
    lower = frame["customer_message"].str.casefold()
    risk += lower.str.contains("refund|charge|bag|baggage|reservation|confirm|account|mile|delay|cancel|change|complaint", regex=True).astype(int) * 2
    selected = frame.assign(_risk=risk).sort_values(["_risk", "customer_tweet_id"], ascending=[False, True]).head(n).copy()
    result = selected[["customer_tweet_id", "customer_message"]].copy()
    result["reviewed_intent"] = ""
    result["reviewed_route"] = ""
    result["review_notes"] = ""
    result.to_csv(output, index=False)
    click.echo(f"Wrote {len(result)} independent-review rows to {output}. Suggestions intentionally excluded.")


@cli.command("promote-reviewed-gold")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), default=EVALUATION_DIR / "golden_review_queue.csv")
@click.option("--output", type=click.Path(path_type=Path), default=EVALUATION_DIR / "golden.csv")
def promote_reviewed_gold(input_path: Path, output: Path) -> None:
    """Freeze reviewed labels as gold, preserving only annotation fields and provenance notes."""
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    required = {"customer_tweet_id", "customer_message", "reviewed_intent", "reviewed_route"}
    missing = required - set(frame.columns)
    if missing:
        raise click.ClickException(f"Missing columns: {sorted(missing)}")
    result = frame[["customer_tweet_id", "customer_message", "reviewed_intent", "reviewed_route"]].rename(columns={
        "reviewed_intent": "gold_intent", "reviewed_route": "gold_route",
    })
    result["annotator_notes"] = frame.get("review_notes", "")
    issues = validate_golden_labels(result)
    if issues:
        raise click.ClickException(" ".join(issues))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    click.echo(f"Wrote {len(result)} reviewed gold labels to {output}")


@cli.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), default=PROCESSED_DIR / "delta_linked.csv")
@click.option("--artifact", type=click.Path(path_type=Path), default=ARTIFACTS_DIR / "delta_agent.joblib")
@click.option("--exclude-ids", type=click.Path(exists=True, path_type=Path), default=None)
def train(input_path: Path, artifact: Path, exclude_ids: Path | None) -> None:
    """Fit the transparent classifier and historical-reply retriever on silver labels."""
    history = pd.read_csv(input_path, dtype=str).fillna("")
    if exclude_ids:
        held_out_ids = set(pd.read_csv(exclude_ids, dtype=str).fillna("")["customer_tweet_id"])
        history = history[~history.customer_tweet_id.isin(held_out_ids)].copy()
    history = label_silver(history)
    model = IntentModel().fit(history.customer_message, history.silver_intent)
    agent = SupportAgent(model, ReplyRetriever().fit(history))
    save_agent(agent, artifact)
    click.echo(f"Trained on {len(history)} silver-labelled examples; saved {artifact}")


@cli.command()
@click.option("--kind", type=click.Choice(["trivial", "keyword"]), required=True)
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def baseline(kind: str, input_path: Path, output: Path) -> None:
    """Run a required intent/routing baseline over a labelled or unlabelled CSV."""
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    result = trivial_predictions(frame) if kind == "trivial" else keyword_predictions(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    click.echo(f"Wrote {kind} baseline predictions to {output}")


@cli.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--artifact", type=click.Path(exists=True, path_type=Path), default=ARTIFACTS_DIR / "delta_agent.joblib")
@click.option("--output", type=click.Path(path_type=Path), required=True)
def predict(input_path: Path, artifact: Path, output: Path) -> None:
    """Add intent, route, draft, and retrieval provenance to a CSV of messages."""
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    if "customer_message" not in frame:
        raise click.ClickException("Input must have a customer_message column.")
    agent = load_agent(artifact)
    outputs = [agent.respond(message) for message in frame.customer_message]
    result = frame.copy()
    for key in ("predicted_intent", "intent_confidence", "decision", "decision_reason", "draft_reply"):
        result[key] = [item[key] for item in outputs]
    result["evidence_json"] = [json.dumps(item["evidence"]) for item in outputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    click.echo(f"Wrote {len(result)} predictions to {output}")


@cli.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=EVALUATION_DIR / "metrics.json")
def evaluate(input_path: Path, output: Path) -> None:
    """Compute headline intent and routing metrics from predictions joined to gold labels."""
    frame = pd.read_csv(input_path).fillna("")
    required = {"gold_intent", "gold_route", "predicted_intent", "decision"}
    missing = required - set(frame.columns)
    if missing:
        raise click.ClickException(f"Missing columns: {sorted(missing)}")
    # A blank template must never produce a plausible-looking score file.
    issues = validate_golden_labels(frame)
    if issues:
        raise click.ClickException(" ".join(issues))
    result = evaluate_predictions(frame, output)
    click.echo(json.dumps(result, indent=2))


@cli.command("judge-replies")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--model", default="gpt-4.1-mini", show_default=True)
def judge_replies(input_path: Path, output: Path, model: str) -> None:
    """Score generated drafts with the documented rubric; requires OPENAI_API_KEY."""
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    required = {"customer_message", "draft_reply", "evidence_json"}
    missing = required - set(frame.columns)
    if missing:
        raise click.ClickException(f"Missing columns: {sorted(missing)}")
    scores = []
    for row in frame.itertuples(index=False):
        verdict = judge_reply(row.customer_message, row.draft_reply, json.loads(row.evidence_json), model)
        scores.append(verdict)
    result = frame.copy()
    result["llm_judge_json"] = [json.dumps(score) for score in scores]
    result["llm_overall_score"] = [score.get("overall_score", "") for score in scores]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    click.echo(f"Wrote {len(result)} LLM-judge scores to {output}")


@cli.command("judge-agreement")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
def judge_agreement(input_path: Path) -> None:
    """Report agreement between blinded human and LLM-judge scores."""
    click.echo(json.dumps(judge_human_agreement(pd.read_csv(input_path)), indent=2))


if __name__ == "__main__":
    cli()
