import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response

from api.dependencies import trigger_deployment
from core.config import (
    PREFECT_TRAIN_DEPLOYMENT,
    PREFECT_EVALUATE_AND_RETRAIN_DEPLOYMENT,
    PREFECT_PREDICT_DEPLOYMENT,
    PREFECT_RETRAIN_DEPLOYMENT,
    REFERENCE_DATA_PATH,
)
from core.schemas import BatchStatus
from monitoring.prometheus import render_metrics
from storage.artifacts import mirror_file_to_object_store
from storage.paths import (
    data_drift_html_path,
    evaluation_output_path,
    prediction_output_path,
    raw_prediction_key,
    raw_prediction_path,
    raw_truth_key,
    raw_truth_path,
    retrain_output_path,
)

app = FastAPI(title="LMS MLOps API")
BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

def _new_batch_id() -> str:
    return uuid4().hex[:12]

def _ensure_csv(file: UploadFile) -> None:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are supported",
        )
    
def _ensure_batch_id(batch_id: str) -> None:
    if not BATCH_ID_PATTERN.fullmatch(batch_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "batch_id must be 1-64 characters and contain only letters, "
                "numbers, dots, underscores, or hyphens"
            ),
        )


async def _save_upload(file: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)


def _artifact_status(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
    }


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics", include_in_schema=False)
def metrics():
    content, media_type = render_metrics()
    return Response(content=content, media_type=media_type)


@app.post("/models/champion/train", status_code=status.HTTP_202_ACCEPTED)
async def train_initial_champion(csv_path: str | None = None):
    train_path = Path(csv_path) if csv_path else REFERENCE_DATA_PATH

    if not train_path.exists():
        raise HTTPException(status_code=404, detail="Training CSV not found")

    flow_run_id = await trigger_deployment(
        name=PREFECT_TRAIN_DEPLOYMENT,
        parameters={
            "csv_path": str(train_path),
        },
    )

    return {
        "status": "accepted",
        "flow_run_id": flow_run_id,
        "training_path": str(train_path),
    }


@app.post("/batches/prediction", status_code=status.HTTP_202_ACCEPTED)
async def upload_prediction_batch(
    file: UploadFile = File(...),
    batch_id: str | None = Form(default=None),
):
    _ensure_csv(file)

    batch_id = batch_id or _new_batch_id()
    _ensure_batch_id(batch_id)

    input_path = raw_prediction_path(batch_id)
    output_path = prediction_output_path(batch_id)

    await _save_upload(file, input_path)
    raw_object = mirror_file_to_object_store(
        input_path,
        raw_prediction_key(batch_id),
        content_type="text/csv",
    )

    flow_run_id = await trigger_deployment(
        name=PREFECT_PREDICT_DEPLOYMENT,
        parameters={
            "input_path": str(input_path),
            "batch_id": batch_id,
            "output_path": str(output_path),
        },
    )

    return {
        "status": "accepted",
        "batch_id": batch_id,
        "flow_run_id": flow_run_id,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "object_store": {
            "raw_prediction": raw_object,
        } if raw_object else {},
    }

@app.post("/batches/truth", status_code=status.HTTP_202_ACCEPTED)
async def upload_truth_batch(
    file: UploadFile = File(...),
    batch_id: str = Form(...),
):
    _ensure_csv(file)
    _ensure_batch_id(batch_id)

    truth_path = raw_truth_path(batch_id)
    prediction_path = prediction_output_path(batch_id)
    evaluation_path = evaluation_output_path(batch_id)
    retrain_path = retrain_output_path(batch_id)

    if not prediction_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Prediction output for this batch_id was not found. "
                "Upload prediction first and wait until the prediction flow completes."
            ),
        )

    await _save_upload(file, truth_path)
    raw_object = mirror_file_to_object_store(
        truth_path,
        raw_truth_key(batch_id),
        content_type="text/csv",
    )

    flow_run_id = await trigger_deployment(
        name=PREFECT_EVALUATE_AND_RETRAIN_DEPLOYMENT,
        parameters={
            "truth_path": str(truth_path),
            "prediction_path": str(prediction_path),
            "batch_id": batch_id,
            "evaluation_path": str(evaluation_path),
            "retrain_output_path": str(retrain_path),
        },
    )

    return {
        "status": "accepted",
        "batch_id": batch_id,
        "flow_run_id": flow_run_id,
        "truth_path": str(truth_path),
        "prediction_path": str(prediction_path),
        "evaluation_path": str(evaluation_path),
        "retrain_path": str(retrain_path),
        "object_store": {
            "raw_truth": raw_object,
        } if raw_object else {},
    }

@app.post("/batches/{batch_id}/retrain", status_code=status.HTTP_202_ACCEPTED)
async def trigger_retrain(batch_id: str):
    _ensure_batch_id(batch_id)

    truth_path = raw_truth_path(batch_id)
    evaluation_path = evaluation_output_path(batch_id)
    retrain_path = retrain_output_path(batch_id)

    if not truth_path.exists():
        raise HTTPException(status_code=404, detail="Truth CSV not found")

    if not evaluation_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation result not found")

    flow_run_id = await trigger_deployment(
        name=PREFECT_RETRAIN_DEPLOYMENT,
        parameters={
            "training_path": str(truth_path),
            "evaluation_path": str(evaluation_path),
            "output_path": str(retrain_path),
        },
    )

    return {
        "status": "accepted",
        "batch_id": batch_id,
        "flow_run_id": flow_run_id,
        "training_path": str(truth_path),
        "evaluation_path": str(evaluation_path),
        "retrain_path": str(retrain_path),
    }


@app.get("/batches/{batch_id}", response_model=BatchStatus)
def get_batch_status(batch_id: str):
    _ensure_batch_id(batch_id)

    prediction_path = prediction_output_path(batch_id)
    evaluation_path = evaluation_output_path(batch_id)
    retrain_path = retrain_output_path(batch_id)

    evaluation = _load_json_if_exists(evaluation_path)
    retrain = _load_json_if_exists(retrain_path)
    known_paths = [
        raw_prediction_path(batch_id),
        prediction_path,
        raw_truth_path(batch_id),
        evaluation_path,
        retrain_path,
    ]

    return {
        "batch_id": batch_id,
        "status": "known" if any(path.exists() for path in known_paths) else "not_found",
        "artifacts": {
            "raw_prediction": _artifact_status(raw_prediction_path(batch_id)),
            "prediction_output": _artifact_status(prediction_path),
            "data_drift_report": _artifact_status(data_drift_html_path(batch_id)),
            "raw_truth": _artifact_status(raw_truth_path(batch_id)),
            "evaluation": _artifact_status(evaluation_path),
            "retrain": _artifact_status(retrain_path),
        },
        "retrain_decision": (
            evaluation.get("retrain_decision") if evaluation is not None else None
        ),
        "promotion_decision": (
            retrain.get("promotion_decision") if retrain is not None else None
        ),
    }


@app.get("/batches/{batch_id}/predictions")
def get_prediction_output(batch_id: str):
    _ensure_batch_id(batch_id)

    path = prediction_output_path(batch_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Prediction output not found")

    return FileResponse(path)


@app.get("/batches/{batch_id}/drift-report")
def get_data_drift_report(batch_id: str):
    _ensure_batch_id(batch_id)

    path = data_drift_html_path(batch_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Data drift report not found")

    return FileResponse(path, media_type="text/html")


@app.get("/batches/{batch_id}/evaluation")
def get_evaluation_output(batch_id: str):
    _ensure_batch_id(batch_id)

    path = evaluation_output_path(batch_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluation output not found")

    return FileResponse(path)

@app.get("/batches/{batch_id}/retrain")
def get_retrain_output(batch_id: str):
    _ensure_batch_id(batch_id)

    path = retrain_output_path(batch_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Retrain output not found")

    return FileResponse(path)
