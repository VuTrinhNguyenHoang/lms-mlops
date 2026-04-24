from pathlib import Path

from core.config import LOCAL_OUTPUT_DIR, LOCAL_REPORT_DIR, LOCAL_STORAGE_DIR


def raw_prediction_path(batch_id: str) -> Path:
    return LOCAL_STORAGE_DIR / "raw" / "prediction" / f"{batch_id}.csv"


def raw_truth_path(batch_id: str) -> Path:
    return LOCAL_STORAGE_DIR / "raw" / "truth" / f"{batch_id}.csv"


def prediction_output_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "predictions" / f"{batch_id}.csv"


def evaluation_output_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "evaluations" / f"{batch_id}.json"


def retrain_output_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "retrain" / f"{batch_id}.json"


def merged_training_dataset_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "training" / "merged" / f"{batch_id}.csv"


def data_drift_html_path(batch_id: str) -> Path:
    return LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.html"


def data_drift_json_path(batch_id: str) -> Path:
    return LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.json"


def data_drift_summary_path(batch_id: str) -> Path:
    return LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.summary.json"
