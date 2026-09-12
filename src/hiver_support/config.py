from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BRAND_HANDLE = "Delta"
SEED = 20260909

INTENTS = (
    "booking_change_or_cancellation",
    "baggage_issue",
    "flight_disruption_or_status",
    "check_in_or_boarding",
    "refund_or_payment",
    "loyalty_or_account",
    "general_information",
    "other_or_ambiguous",
)
