from pathlib import Path
import json

from prefect import flow, get_run_logger
from core.config import LOCAL_REPORT_DIR, REFERENCE_DATA_PATH
from data.loading import save_csv, load_csv
from drift.data_drift import compute_data_drift
from monitoring.prometheus import record_prediction_metrics
from models.predict import predict_with_champion

@flow(name="predict-batch", log_prints=True)
def predict_batch_flow(
    input_path: str,
    batch_id: str,
    output_path: str | None = None,
) -> dict:
    logger = get_run_logger()

    if output_path is None:
        output_path = f"outputs/predictions/{batch_id}.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting prediction for batch_id=%s from %s", batch_id, input_path)

    predictions = predict_with_champion(
        csv_path=input_path,
        batch_id=batch_id,
    )

    save_csv(predictions, output_path)

    reference_df = load_csv(REFERENCE_DATA_PATH)
    current_df = load_csv(input_path)

    reference_predictions = predict_with_champion(
        csv_path=str(REFERENCE_DATA_PATH),
        batch_id=f"{batch_id}-reference",
    )

    drift_html_path = (
        LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.html"
    )
    drift_json_path = (
        LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.json"
    )
    drift_summary_path = (
        LOCAL_REPORT_DIR / "evidently" / "data_drift" / f"{batch_id}.summary.json"
    )

    drift_metrics = compute_data_drift(
        reference_df=reference_df,
        current_df=current_df,
        current_predictions=predictions,
        reference_predictions=reference_predictions,
        html_path=str(drift_html_path),
        json_path=str(drift_json_path),
    )

    drift_summary_path.parent.mkdir(parents=True, exist_ok=True)
    drift_summary_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "input_path": input_path,
                "reference_path": str(REFERENCE_DATA_PATH),
                "drift_html_path": str(drift_html_path),
                "drift_json_path": str(drift_json_path),
                "metrics": drift_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model_version = str(predictions["model_version"].iloc[0])

    logger.info(
        "Saved %s prediction rows to %s using model version %s",
        len(predictions),
        output_path,
        model_version,
    )

    summary = {
        "batch_id": batch_id,
        "input_path": input_path,
        "output_path": output_path,
        "row_count": len(predictions),
        "model_name": predictions["model_name"].iloc[0],
        "model_version": model_version,
        "drift_html_path": str(drift_html_path),
        "drift_json_path": str(drift_json_path),
        "drift_summary_path": str(drift_summary_path),
        "drift_metrics": drift_metrics,
    }

    record_prediction_metrics(summary)
    return summary


if __name__ == "__main__":
    print(
        predict_batch_flow(
            input_path="simulated_data.csv",
            batch_id="demo-batch",
        )
    )
