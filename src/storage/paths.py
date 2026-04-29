from pathlib import Path

from core.config import LOCAL_OUTPUT_DIR, LOCAL_REPORT_DIR, LOCAL_STORAGE_DIR


def _csv_key(prefix: str, batch_id: str) -> str:
    return f"{prefix}/{batch_id}.csv"


def _json_key(prefix: str, batch_id: str) -> str:
    return f"{prefix}/{batch_id}.json"


def raw_prediction_path(batch_id: str) -> Path:
    return LOCAL_STORAGE_DIR / "raw" / "prediction" / f"{batch_id}.csv"


def raw_prediction_key(batch_id: str) -> str:
    return _csv_key("raw/prediction", batch_id)


def raw_truth_path(batch_id: str) -> Path:
    return LOCAL_STORAGE_DIR / "raw" / "truth" / f"{batch_id}.csv"


def raw_truth_key(batch_id: str) -> str:
    return _csv_key("raw/truth", batch_id)


def prediction_output_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "predictions" / f"{batch_id}.csv"


def prediction_output_key(batch_id: str) -> str:
    return _csv_key("processed/predictions", batch_id)


def evaluation_output_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "evaluations" / f"{batch_id}.json"


def evaluation_output_key(batch_id: str) -> str:
    return _json_key("processed/evaluations", batch_id)


def retrain_output_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "retrain" / f"{batch_id}.json"


def retrain_output_key(batch_id: str) -> str:
    return _json_key("decisions/retrain", batch_id)


def merged_training_dataset_path(batch_id: str) -> Path:
    return LOCAL_OUTPUT_DIR / "training" / "merged" / f"{batch_id}.csv"


def merged_training_dataset_key(batch_id: str) -> str:
    return _csv_key("training/merged", batch_id)


def data_drift_html_path(batch_id: str) -> Path:
    return LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.html"


def data_drift_html_key(batch_id: str) -> str:
    return f"reports/evidently/data_drift/{batch_id}.html"


def data_drift_json_path(batch_id: str) -> Path:
    return LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.json"


def data_drift_json_key(batch_id: str) -> str:
    return _json_key("reports/evidently/data_drift", batch_id)


def data_drift_summary_path(batch_id: str) -> Path:
    return LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.summary.json"


def data_drift_summary_key(batch_id: str) -> str:
    return f"reports/evidently/data_drift/{batch_id}.summary.json"
