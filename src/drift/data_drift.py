from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from core.config import DATA_DRIFT_SHARE_THRESHOLD, PREDICTION_SCORE_DRIFT_THRESHOLD
from core.contracts import FEATURE_COLUMNS

def _select_features(df):
    return df[FEATURE_COLUMNS].copy()

def _find_drifted_columns_metric(snapshot_dict: dict) -> dict:
    for metric in snapshot_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        config_type = metric.get("config", {}).get("type", "")

        if "DriftedColumnsCount" in metric_name or config_type.endswith("DriftedColumnsCount"):
            return metric

    raise ValueError("Evidently report does not contain DriftedColumnsCount metric")

def _prediction_score_summary(current_predictions, reference_predictions=None) -> dict:
    current_mean = float(current_predictions["risk_score"].mean())

    if reference_predictions is None:
        return {
            "lms_prediction_score_mean": current_mean,
            "lms_reference_prediction_score_mean": None,
            "lms_prediction_score_drift": None,
            "prediction_score_drift_detected": None,
        }

    reference_mean = float(reference_predictions["risk_score"].mean())
    score_drift = abs(current_mean - reference_mean)

    return {
        "lms_prediction_score_mean": current_mean,
        "lms_reference_prediction_score_mean": reference_mean,
        "lms_prediction_score_drift": score_drift,
        "prediction_score_drift_detected": score_drift > PREDICTION_SCORE_DRIFT_THRESHOLD,
    }

def compute_data_drift(
    reference_df,
    current_df,
    current_predictions=None,
    reference_predictions=None,
    html_path: str | None = None,
    json_path: str | None = None,
) -> dict:
    reference_features = _select_features(reference_df)
    current_features = _select_features(current_df)

    report = Report(
        [
            DataDriftPreset(drift_share=DATA_DRIFT_SHARE_THRESHOLD),
        ]
    )

    snapshot = report.run(current_features, reference_features)

    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(html_path)

    if json_path is not None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_json(json_path)

    snapshot_dict = snapshot.dict()
    drift_metric = _find_drifted_columns_metric(snapshot_dict)
    drift_value = drift_metric["value"]

    drifted_feature_count = int(drift_value["count"])
    drift_share = float(drift_value["share"])
    total_feature_count = len(FEATURE_COLUMNS)

    metrics = {
        "lms_data_drift_share": drift_share,
        "lms_drifted_feature_count": drifted_feature_count,
        "lms_total_feature_count": total_feature_count,
        "data_drift_detected": drift_share > DATA_DRIFT_SHARE_THRESHOLD,
    }

    if current_predictions is not None:
        metrics.update(
            _prediction_score_summary(
                current_predictions=current_predictions,
                reference_predictions=reference_predictions,
            )
        )

    return metrics
