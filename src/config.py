"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_LABELS_CSV = DATA_DIR / "test_labels.csv"
MODEL_PATH = ROOT / "models" / "toxic_comment_model.joblib"
REPORT_DIR = ROOT / "reports"

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_NAMES = {
    "toxic": "Toxic", "severe_toxic": "Severe toxic", "obscene": "Obscene",
    "threat": "Threat", "insult": "Insult", "identity_hate": "Identity hate",
}
SEED = 42
