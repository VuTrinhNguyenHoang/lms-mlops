from prefect import flow, get_run_logger

from flows.retrain_flow import retrain_flow
from flows.truth_flow import evaluate_truth_flow


@flow(name="evaluate-and-maybe-retrain", log_prints=True)
def evaluate_and_maybe_retrain_flow(
    truth_path: str,
    prediction_path: str,
    batch_id: str,
    evaluation_path: str | None = None,
    retrain_output_path: str | None = None,
    reference_path: str | None = None,
    merged_training_path: str | None = None,
) -> dict:
    logger = get_run_logger()
    logger.info("Starting evaluate-and-maybe-retrain for batch_id=%s", batch_id)

    evaluation = evaluate_truth_flow(
        truth_path=truth_path,
        prediction_path=prediction_path,
        batch_id=batch_id,
        output_path=evaluation_path,
    )

    summary = {
        "batch_id": batch_id,
        "truth_path": truth_path,
        "prediction_path": prediction_path,
        "evaluation_path": evaluation["output_path"],
        "retrain_decision": evaluation.get("retrain_decision", False),
        "evaluation": evaluation,
        "retrain": None,
    }

    if not evaluation.get("retrain_decision", False):
        logger.info("Retrain not required for batch_id=%s", batch_id)
        return summary

    retrain = retrain_flow(
        training_path=truth_path,
        evaluation_path=evaluation["output_path"],
        output_path=retrain_output_path,
        reference_path=reference_path,
        merged_training_path=merged_training_path,
    )

    summary["retrain"] = retrain
    summary["retrain_path"] = retrain["output_path"]
    return summary
