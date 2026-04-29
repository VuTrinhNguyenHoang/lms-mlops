import json
from pathlib import Path

from prefect import flow, get_run_logger

from core.config import REFERENCE_DATA_PATH
from data.retrain_dataset import build_retrain_dataset
from monitoring.prometheus import record_retrain_metrics
from models.registry import register_model_version, set_champion_alias
from models.train import train_and_log_candidates
from rules.promotion import should_promote
from storage.artifacts import compact_artifact_map, mirror_file_to_object_store
from storage.paths import (
    evaluation_output_path,
    merged_training_dataset_key,
    merged_training_dataset_path,
    retrain_output_key,
    retrain_output_path,
)

def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _write_json(payload: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

def _candidate_metrics(results: dict) -> dict:
    return {
        name: item["metrics"]
        for name, item in results.items()
    }

def _candidate_runs(results: dict) -> dict:
    return {
        name: item["mlflow"]
        for name, item in results.items()
    }

@flow(name="retrain-model", log_prints=True)
def retrain_flow(
    training_path: str,
    evaluation_path: str,
    output_path: str | None = None,
    reference_path: str | None = None,
    merged_training_path: str | None = None,
) -> dict:
    logger = get_run_logger()

    evaluation = _load_json(evaluation_path)
    batch_id = evaluation["batch_id"]
    # Kept as training_path for the existing deployment/API contract.
    # At this stage it represents the newest truth batch path.
    truth_paths = [training_path]

    if output_path is None:
        output_path = str(retrain_output_path(batch_id))

    if not evaluation.get("retrain_decision", False):
        summary = {
            "batch_id": batch_id,
            "training_path": training_path,
            "truth_paths": truth_paths,
            "evaluation_path": evaluation_path,
            "output_path": output_path,
            "status": "skipped",
            "retrain_decision": False,
            "retrain_reasons": evaluation.get("retrain_reasons", []),
            "promotion_decision": False,
            "promotion_reasons": ["retrain_decision=False"],
        }

        _write_json(summary, output_path)
        retrain_object = mirror_file_to_object_store(
            output_path,
            retrain_output_key(batch_id),
            content_type="application/json",
        )
        summary["object_store"] = compact_artifact_map(
            {
                "retrain_decision": retrain_object,
            }
        )
        _write_json(summary, output_path)
        if retrain_object is not None:
            mirror_file_to_object_store(
                output_path,
                retrain_output_key(batch_id),
                content_type="application/json",
            )
        record_retrain_metrics(summary)
        logger.info("Retrain skipped for batch_id=%s", batch_id)
        return summary

    reference_path = reference_path or str(REFERENCE_DATA_PATH)
    merged_training_path = merged_training_path or str(
        merged_training_dataset_path(batch_id)
    )
    training_dataset = build_retrain_dataset(
        reference_path=reference_path,
        truth_paths=truth_paths,
        batch_id=batch_id,
        output_path=merged_training_path,
    )
    merged_training_object = mirror_file_to_object_store(
        training_dataset["output_path"],
        merged_training_dataset_key(batch_id),
        content_type="text/csv",
    )

    logger.info(
        "Starting retrain for batch_id=%s from merged dataset %s",
        batch_id,
        training_dataset["output_path"],
    )

    train_result = train_and_log_candidates(
        csv_path=training_dataset["output_path"],
        include_champion_baseline=True,
    )

    best_model_name = train_result["best_model_name"]
    best_candidate = train_result["results"][best_model_name]

    promotion_decision, promotion_reasons = should_promote(
        candidate_metrics=best_candidate["metrics"],
        current_metrics=train_result["champion_metrics"],
    )

    registered_model_name = None
    model_version = None

    if promotion_decision:
        version = register_model_version(best_candidate["mlflow"]["model_uri"])
        set_champion_alias(version.version)

        registered_model_name = version.name
        model_version = version.version

        logger.info(
            "Promoted retrained model %s version %s",
            registered_model_name,
            model_version,
        )
    else:
        logger.info("Retrained candidate was not promoted for batch_id=%s", batch_id)

    summary = {
        "batch_id": batch_id,
        "training_path": training_dataset["output_path"],
        "truth_paths": truth_paths,
        "training_dataset": training_dataset,
        "evaluation_path": evaluation_path,
        "output_path": output_path,
        "status": "completed",
        "retrain_decision": True,
        "retrain_reasons": evaluation.get("retrain_reasons", []),
        "best_model_name": best_model_name,
        "candidate_metrics": best_candidate["metrics"],
        "candidate_metrics_by_model": _candidate_metrics(train_result["results"]),
        "candidate_runs": _candidate_runs(train_result["results"]),
        "champion_metrics": train_result["champion_metrics"],
        "promotion_decision": promotion_decision,
        "promotion_reasons": promotion_reasons,
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "champion_alias": "champion" if promotion_decision else None,
        "object_store": compact_artifact_map(
            {
                "merged_training_dataset": merged_training_object,
            }
        ),
    }

    _write_json(summary, output_path)
    retrain_object = mirror_file_to_object_store(
        output_path,
        retrain_output_key(batch_id),
        content_type="application/json",
    )
    summary["object_store"] = compact_artifact_map(
        {
            **summary["object_store"],
            "retrain_decision": retrain_object,
        }
    )
    _write_json(summary, output_path)
    if retrain_object is not None:
        mirror_file_to_object_store(
            output_path,
            retrain_output_key(batch_id),
            content_type="application/json",
        )
    record_retrain_metrics(summary)
    return summary

if __name__ == "__main__":
    print(
        retrain_flow(
            training_path=str(REFERENCE_DATA_PATH),
            evaluation_path=str(evaluation_output_path("demo-batch")),
        )
    )
