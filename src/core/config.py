from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MLflow
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")
MLFLOW_REGISTRY_URI = MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME = "lms-dropout-risk"
MLFLOW_REGISTERED_MODEL_NAME = "lms-dropout-risk-model"
MLFLOW_CHAMPION_ALIAS = "champion"
MLFLOW_MODEL_ARTIFACT_NAME = "model"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2

RISK_THRESHOLD = 0.5

RISK_LEVEL_LOW = 0.3
RISK_LEVEL_HIGH = 0.7

PRIMARY_METRIC = "f1_risk"

def risk_level(score: float) -> str:
    if score >= RISK_LEVEL_HIGH:
        return "high"
    if score >= RISK_LEVEL_LOW:
        return "medium"
    return "low"

# Local paths
LOCAL_STORAGE_DIR = PROJECT_ROOT / "storage"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Prefect deployments
PREFECT_TRAIN_DEPLOYMENT = "train-initial-champion/train-initial-champion"
PREFECT_PREDICT_DEPLOYMENT = "predict-batch/predict-batch"
PREFECT_TRUTH_DEPLOYMENT = "evaluate-truth/evaluate-truth"
PREFECT_RETRAIN_DEPLOYMENT = "retrain-model/retrain-model"

# Drift
REFERENCE_DATA_PATH = PROJECT_ROOT / "simulated_data.csv"
LOCAL_REPORT_DIR = LOCAL_OUTPUT_DIR / "reports"

DATA_DRIFT_SHARE_THRESHOLD = 0.5
PREDICTION_SCORE_DRIFT_THRESHOLD = 0.05