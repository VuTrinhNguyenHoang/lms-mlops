from pathlib import Path

from prefect import flow, get_run_logger

from data.loading import save_csv
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

    model_version = str(predictions["model_version"].iloc[0])

    logger.info(
        "Saved %s prediction rows to %s using model version %s",
        len(predictions),
        output_path,
        model_version,
    )

    return {
        "batch_id": batch_id,
        "input_path": input_path,
        "output_path": output_path,
        "row_count": len(predictions),
        "model_name": predictions["model_name"].iloc[0],
        "model_version": model_version,
    }


if __name__ == "__main__":
    print(
        predict_batch_flow(
            input_path="simulated_data.csv",
            batch_id="demo-batch",
        )
    )
