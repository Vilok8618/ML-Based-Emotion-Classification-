from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DEFAULT_MODEL_PATH = MODEL_DIR / "emotion_model.joblib"

# Labels of the dair-ai/emotion dataset
HF_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
RANDOM_STATE = 42
