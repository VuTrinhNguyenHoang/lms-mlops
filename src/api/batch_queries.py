import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import LOCAL_OUTPUT_DIR, LOCAL_REPORT_DIR, LOCAL_STORAGE_DIR
from core.contracts import ID_COLUMNS
from storage.paths import (
    evaluation_output_path,
    prediction_output_path,
    raw_prediction_path,
    raw_truth_path,
    retrain_output_path,
)


UI_STATUSES = {
    "PROCESSING_PREDICTION": "processing_prediction",
    "PREDICTED": "predicted",
    "EVALUATING": "evaluating",
    "EVALUATED": "evaluated",
    "NOT_FOUND": "not_found",
}

USER_METRIC_KEYS = [
    "accuracy",
    "precision_risk",
    "recall_risk",
    "f1_risk",
    "false_negative_count",
]


def list_batch_ids() -> list[str]:
    batch_ids: set[str] = set()

    for directory, suffix in [
        (LOCAL_STORAGE_DIR / "raw" / "prediction", ".csv"),
        (LOCAL_STORAGE_DIR / "raw" / "truth", ".csv"),
        (LOCAL_OUTPUT_DIR / "predictions", ".csv"),
        (LOCAL_OUTPUT_DIR / "evaluations", ".json"),
        (LOCAL_OUTPUT_DIR / "retrain", ".json"),
    ]:
        batch_ids.update(_ids_from_directory(directory, suffix))

    drift_dir = LOCAL_REPORT_DIR / "evidently" / "data_drift"
    batch_ids.update(_ids_from_directory(drift_dir, ".html"))
    batch_ids.update(_ids_from_directory(drift_dir, ".json"))
    batch_ids.update(
        path.name.removesuffix(".summary.json")
        for path in _safe_glob(drift_dir, "*.summary.json")
    )

    return sorted(batch_ids)


def get_batch_user_status(batch_id: str) -> str:
    raw_prediction_exists = raw_prediction_path(batch_id).exists()
    prediction_exists = prediction_output_path(batch_id).exists()
    truth_exists = raw_truth_path(batch_id).exists()
    evaluation_exists = evaluation_output_path(batch_id).exists()
    retrain_exists = retrain_output_path(batch_id).exists()

    if evaluation_exists:
        return UI_STATUSES["EVALUATED"]
    if truth_exists:
        return UI_STATUSES["EVALUATING"]
    if prediction_exists:
        return UI_STATUSES["PREDICTED"]
    if raw_prediction_exists:
        return UI_STATUSES["PROCESSING_PREDICTION"]
    if retrain_exists:
        return UI_STATUSES["EVALUATED"]
    return UI_STATUSES["NOT_FOUND"]


def get_batch_summary(batch_id: str) -> dict[str, Any]:
    distribution = _risk_distribution(batch_id)

    return {
        "batch_id": batch_id,
        "status": get_batch_user_status(batch_id),
        "total_students": sum(distribution.values()),
        "risk_distribution": distribution,
        "truth_uploaded": raw_truth_path(batch_id).exists(),
        "evaluated": evaluation_output_path(batch_id).exists(),
    }


def list_batches() -> dict[str, list[dict[str, Any]]]:
    items = []

    for batch_id in list_batch_ids():
        summary = get_batch_summary(batch_id)
        distribution = summary["risk_distribution"]
        items.append(
            {
                "batch_id": batch_id,
                "status": summary["status"],
                "total_students": summary["total_students"],
                "high_risk_count": distribution["high"],
                "medium_risk_count": distribution["medium"],
                "low_risk_count": distribution["low"],
                "truth_uploaded": summary["truth_uploaded"],
                "evaluated": summary["evaluated"],
                "created_at": _created_at(batch_id),
                "updated_at": _updated_at(batch_id),
            }
        )

    items.sort(key=lambda item: item["created_at"] or "", reverse=True)
    return {"items": items}


def get_prediction_preview(
    batch_id: str,
    page: int = 1,
    page_size: int = 20,
    risk_level: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    path = prediction_output_path(batch_id)
    if not path.exists():
        raise FileNotFoundError("Prediction output not found")

    df = pd.read_csv(path)
    df = df[df["batch_id"].astype(str) == batch_id].copy()

    if risk_level:
        df = df[df["risk_level"].astype(str).str.lower() == risk_level.lower()]

    if q:
        query = q.strip().lower()
        if query:
            mask = pd.Series(False, index=df.index)
            for column in ID_COLUMNS:
                mask = mask | df[column].astype(str).str.lower().str.contains(
                    query,
                    regex=False,
                    na=False,
                )
            df = df[mask]

    if "risk_score" in df.columns:
        df = df.sort_values("risk_score", ascending=False)

    total = int(len(df))
    start = (page - 1) * page_size
    stop = start + page_size

    items = []
    for _, row in df.iloc[start:stop].iterrows():
        item = {
            "id": _json_value(row[ID_COLUMNS[0]]),
            "risk_score": float(row["risk_score"]),
            "predicted_label": int(row["predicted_label"]),
            "risk_level": str(row["risk_level"]),
        }
        items.append(item)

    return {
        "batch_id": batch_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
    }


def get_evaluation_summary(batch_id: str) -> dict[str, Any]:
    path = evaluation_output_path(batch_id)
    if not path.exists():
        raise FileNotFoundError("Evaluation output not found")

    evaluation = _load_json(path)
    metrics = evaluation.get("metrics", {})

    return {
        "batch_id": batch_id,
        "truth_rows": int(evaluation.get("truth_rows", 0)),
        "matched_rows": int(evaluation.get("matched_rows", 0)),
        "matched_ratio": float(evaluation.get("matched_ratio", 0)),
        "metrics": {
            key: metrics.get(key)
            for key in USER_METRIC_KEYS
            if key in metrics
        },
    }


def _ids_from_directory(directory: Path, suffix: str) -> set[str]:
    return {
        path.name.removesuffix(suffix)
        for path in _safe_glob(directory, f"*{suffix}")
        if path.name.endswith(suffix)
        and not path.name.endswith(".summary.json")
    }


def _safe_glob(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    return list(directory.glob(pattern))


def _risk_distribution(batch_id: str) -> dict[str, int]:
    path = prediction_output_path(batch_id)
    if not path.exists():
        return {"high": 0, "medium": 0, "low": 0}

    try:
        df = pd.read_csv(path)
        if "batch_id" in df.columns:
            df = df[df["batch_id"].astype(str) == batch_id]
        if "risk_level" not in df.columns:
            return {"high": 0, "medium": 0, "low": 0}
    except Exception:
        return {"high": 0, "medium": 0, "low": 0}

    counts = df["risk_level"].astype(str).str.lower().value_counts()
    return {
        "high": int(counts.get("high", 0)),
        "medium": int(counts.get("medium", 0)),
        "low": int(counts.get("low", 0)),
    }


def _created_at(batch_id: str) -> str | None:
    existing_paths = _batch_artifact_paths(batch_id)
    if not existing_paths:
        return None

    timestamp = min(path.stat().st_mtime for path in existing_paths)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _updated_at(batch_id: str) -> str | None:
    existing_paths = _batch_artifact_paths(batch_id)
    if not existing_paths:
        return None

    timestamp = max(path.stat().st_mtime for path in existing_paths)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _batch_artifact_paths(batch_id: str) -> list[Path]:
    paths = [
        raw_prediction_path(batch_id),
        prediction_output_path(batch_id),
        raw_truth_path(batch_id),
        evaluation_output_path(batch_id),
    ]
    return [path for path in paths if path.exists()]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
