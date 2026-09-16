from pathlib import Path

APP_VERSION = "0.1.0"

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "real_estate_data.csv"

MODELS_DIR = BASE_DIR.parent / "models"

MODEL_PATH = MODELS_DIR / "final_model.pkl"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
DASHBOARD_SAMPLE_PATH = MODELS_DIR / "dashboard_sample.csv"

# Для дашборда достаточно 1000 строк.
DASHBOARD_SAMPLE_SIZE = 999999
