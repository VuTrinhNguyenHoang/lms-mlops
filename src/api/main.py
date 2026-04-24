from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response

from api.dependencies import trigger_deployment
from core.config import (
    LOCAL_OUTPUT_DIR,
    LOCAL_STORAGE_DIR,
    PREFECT_PREDICT_DEPLOYMENT,
    PREFECT_RETRAIN_DEPLOYMENT,
    PREFECT_TRUTH_DEPLOYMENT,
)
from monitoring.prometheus import render_metrics

app = FastAPI(title="LMS MLOps API")

def _new_batch_id() -> str:
    return uuid4().hex[:12]

def _ensure_csv(file: UploadFile) -> None:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are supported",
        )
    
async def _save_upload(file: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics", include_in_schema=False)
def metrics():
    content, media_type = render_metrics()
    return Response(content=content, media_type=media_type)

@app.post("/batches/prediction", status_code=status.HTTP_202_ACCEPTED)
async def upload_prediction_batch(
    file: UploadFile = File(...),
    batch_id: str | None = Form(default=None),
):
    _ensure_csv(file)

    batch_id = batch_id or _new_batch_id()
    input_path = LOCAL_STORAGE_DIR / "raw" / "prediction" / f"{batch_id}.csv"
    output_path = LOCAL_OUTPUT_DIR / "predictions" / f"{batch_id}.csv"

    await _save_upload(file, input_path)

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
    }

@app.post("/batches/truth", status_code=status.HTTP_202_ACCEPTED)
async def upload_truth_batch(
    file: UploadFile = File(...),
    batch_id: str = Form(...),
):
    _ensure_csv(file)

    truth_path = LOCAL_STORAGE_DIR / "raw" / "truth" / f"{batch_id}.csv"
    prediction_path = LOCAL_OUTPUT_DIR / "predictions" / f"{batch_id}.csv"
    evaluation_path = LOCAL_OUTPUT_DIR / "evaluations" / f"{batch_id}.json"

    await _save_upload(file, truth_path)

    flow_run_id = await trigger_deployment(
        name=PREFECT_TRUTH_DEPLOYMENT,
        parameters={
            "truth_path": str(truth_path),
            "prediction_path": str(prediction_path),
            "batch_id": batch_id,
            "output_path": str(evaluation_path),
        },
    )

    return {
        "status": "accepted",
        "batch_id": batch_id,
        "flow_run_id": flow_run_id,
        "truth_path": str(truth_path),
        "prediction_path": str(prediction_path),
        "evaluation_path": str(evaluation_path),
    }

@app.post("/batches/{batch_id}/retrain", status_code=status.HTTP_202_ACCEPTED)
async def trigger_retrain(batch_id: str):
    truth_path = LOCAL_STORAGE_DIR / "raw" / "truth" / f"{batch_id}.csv"
    evaluation_path = LOCAL_OUTPUT_DIR / "evaluations" / f"{batch_id}.json"
    retrain_path = LOCAL_OUTPUT_DIR / "retrain" / f"{batch_id}.json"

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

@app.get("/batches/{batch_id}/predictions")
def get_prediction_output(batch_id: str):
    path = LOCAL_OUTPUT_DIR / "predictions" / f"{batch_id}.csv"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Prediction output not found")

    return FileResponse(path)

@app.get("/batches/{batch_id}/evaluation")
def get_evaluation_output(batch_id: str):
    path = LOCAL_OUTPUT_DIR / "evaluations" / f"{batch_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluation output not found")

    return FileResponse(path)

@app.get("/batches/{batch_id}/retrain")
def get_retrain_output(batch_id: str):
    path = LOCAL_OUTPUT_DIR / "retrain" / f"{batch_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Retrain output not found")

    return FileResponse(path)
