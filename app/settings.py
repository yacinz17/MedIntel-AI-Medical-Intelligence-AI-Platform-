from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
DIABETES_MODEL_PATH = MODEL_DIR / "diabetes_rf.joblib"
XRAY_MODEL_PATH = MODEL_DIR / "chest_xray_effnet.keras"

DISCLAIMER = (
    "Research and portfolio prototype only. This software is not a medical device, "
    "does not provide a diagnosis, and must not be used for clinical decisions."
)
