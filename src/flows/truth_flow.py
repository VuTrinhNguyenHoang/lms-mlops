import json
from pathlib import Path

from prefect import flow, get_run_logger

from core.config import LOCAL_REPORT_DIR
from core.contracts import ID_COLUMNS
from data.loading import load_csv
from data.validation import validate_prediction_output_df, validate_truth_df
from models.evaluate import evaluate_prediction_output
from monitoring.prometheus import record_truth_metrics
from rules.retrain import should_retrain
from drift.performance_drift import compute_performance_drift

def _join_truth_and_predictions(truth_df, predictions_df, batch_id: str):
    batch_predictions = predictions_df[predictions_df["batch_id"] == batch_id].copy()

    if batch_predictions.empty:
        raise ValueError(f"No predictions found for batch_id={batch_id}")

    joined = truth_df.merge(
        batch_predictions,
        on=ID_COLUMNS,
        how="inner",
    )

    if joined.empty:
        raise ValueError("Truth and predictions have no matching ID rows")

    return joined, batch_predictions

def _data_drift_summary_path(batch_id: str) -> Path:
    return LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.summary.json"


def _load_data_drift_metrics(batch_id: str) -> tuple[dict | None, str | None]:
    path = _data_drift_summary_path(batch_id)

    if not path.exists():
        return None, None

    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary.get("metrics", {}), str(path)

@flow(name="evaluate-truth", log_prints=True)
def evaluate_truth_flow(
    truth_path: str,
    prediction_path: str,
    batch_id: str,
    output_path: str | None = None,
) -> dict:
    logger = get_run_logger()

    if output_path is None:
        output_path = f"outputs/evaluations/{batch_id}.json"

    logger.info("Evaluating truth for batch_id=%s", batch_id)

    truth_df = validate_truth_df(load_csv(truth_path))
    predictions_df = validate_prediction_output_df(load_csv(prediction_path))

    joined_df, batch_predictions = _join_truth_and_predictions(
        truth_df=truth_df,
        predictions_df=predictions_df,
        batch_id=batch_id,
    )

    metrics = evaluate_prediction_output(joined_df)
    performance_drift = compute_performance_drift(metrics)
    matched_ratio = len(joined_df) / len(truth_df) if len(truth_df) else 0
    data_drift, data_drift_summary_path = _load_data_drift_metrics(batch_id)

    summary = {
        "batch_id": batch_id,
        "truth_path": truth_path,
        "prediction_path": prediction_path,
        "output_path": output_path,
        "truth_rows": int(len(truth_df)),
        "prediction_rows": int(len(batch_predictions)),
        "matched_rows": int(len(joined_df)),
        "matched_ratio": float(matched_ratio),
        "unmatched_truth_rows": int(len(truth_df) - len(joined_df)),
        "metrics": metrics,
        "performance_drift": performance_drift
    }

    if data_drift is not None:
        summary["data_drift"] = data_drift
        summary["data_drift_summary_path"] = data_drift_summary_path

    retrain_decision, retrain_reasons = should_retrain(summary)
    summary["retrain_decision"] = retrain_decision
    summary["retrain_reasons"] = retrain_reasons

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    record_truth_metrics(summary)

    logger.info(
        "Evaluation done for batch_id=%s: matched_rows=%s, f1_risk=%.4f, recall_risk=%.4f",
        batch_id,
        summary["matched_rows"],
        metrics["f1_risk"],
        metrics["recall_risk"],
    )

    return summary

if __name__ == "__main__":
    print(
        evaluate_truth_flow(
            truth_path="simulated_data.csv",
            prediction_path="outputs/predictions/demo-batch.csv",
            batch_id="demo-batch",
        )
    )
