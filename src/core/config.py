import os
from pathlib import Path


def _get_env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def _get_env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_ROOT = _get_env_path(
    "PROJECT_ROOT",
    Path(__file__).resolve().parents[2],
)

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "lms-dropout-risk")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "lms-dropout-risk-model",
)
MLFLOW_CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
MLFLOW_MODEL_ARTIFACT_NAME = os.getenv("MLFLOW_MODEL_ARTIFACT_NAME", "model")

# Training
RANDOM_STATE = _get_env_int("RANDOM_STATE", 42)
TEST_SIZE = _get_env_float("TEST_SIZE", 0.2)

RISK_THRESHOLD = _get_env_float("RISK_THRESHOLD", 0.5)

RISK_LEVEL_LOW = _get_env_float("RISK_LEVEL_LOW", 0.3)
RISK_LEVEL_HIGH = _get_env_float("RISK_LEVEL_HIGH", 0.7)

PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "f1_risk")

def risk_level(score: float) -> str:
    if score >= RISK_LEVEL_HIGH:
        return "high"
    if score >= RISK_LEVEL_LOW:
        return "medium"
    return "low"

# Local paths
LOCAL_STORAGE_DIR = _get_env_path("LOCAL_STORAGE_DIR", PROJECT_ROOT / "storage")
LOCAL_OUTPUT_DIR = _get_env_path("LOCAL_OUTPUT_DIR", PROJECT_ROOT / "outputs")
LOCAL_METRICS_DIR = LOCAL_OUTPUT_DIR / "metrics"

# Object storage
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "lms-mlops")
MINIO_SECURE = _get_env_bool("MINIO_SECURE", False)
MINIO_MIRROR_ARTIFACTS = _get_env_bool(
    "MINIO_MIRROR_ARTIFACTS",
    STORAGE_BACKEND == "minio",
)

# Prefect deployments
PREFECT_TRAIN_DEPLOYMENT = os.getenv(
    "PREFECT_TRAIN_DEPLOYMENT",
    "train-initial-champion/train-initial-champion",
)
PREFECT_PREDICT_DEPLOYMENT = os.getenv(
    "PREFECT_PREDICT_DEPLOYMENT",
    "predict-batch/predict-batch",
)
PREFECT_TRUTH_DEPLOYMENT = os.getenv(
    "PREFECT_TRUTH_DEPLOYMENT",
    "evaluate-truth/evaluate-truth",
)
PREFECT_EVALUATE_AND_RETRAIN_DEPLOYMENT = os.getenv(
    "PREFECT_EVALUATE_AND_RETRAIN_DEPLOYMENT",
    "evaluate-and-maybe-retrain/evaluate-and-maybe-retrain",
)
PREFECT_RETRAIN_DEPLOYMENT = os.getenv(
    "PREFECT_RETRAIN_DEPLOYMENT",
    "retrain-model/retrain-model",
)

# Drift
REFERENCE_DATA_PATH = _get_env_path(
    "REFERENCE_DATA_PATH",
    PROJECT_ROOT / "data" / "reference" / "simulated_data.csv",
)
LOCAL_REPORT_DIR = LOCAL_OUTPUT_DIR / "reports"

DATA_DRIFT_SHARE_THRESHOLD = _get_env_float("DATA_DRIFT_SHARE_THRESHOLD", 0.5)
PREDICTION_SCORE_DRIFT_THRESHOLD = _get_env_float(
    "PREDICTION_SCORE_DRIFT_THRESHOLD",
    0.05,
)
