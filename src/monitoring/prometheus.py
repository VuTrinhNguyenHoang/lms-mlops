import json
from collections import Counter
from pathlib import Path
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter as PromCounter
from prometheus_client import Gauge, generate_latest
from prometheus_client import disable_created_metrics

from core.config import LOCAL_METRICS_DIR


EVENT_LOG_PATH = LOCAL_METRICS_DIR / "events.jsonl"
LATEST_PREDICTION_PATH = LOCAL_METRICS_DIR / "latest_prediction.json"
LATEST_TRUTH_PATH = LOCAL_METRICS_DIR / "latest_truth.json"
LATEST_RETRAIN_PATH = LOCAL_METRICS_DIR / "latest_retrain.json"

disable_created_metrics()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_event(event_type: str, payload: dict) -> None:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_type": event_type,
        "batch_id": payload.get("batch_id"),
        "status": payload.get("status", "completed"),
    }

    with EVENT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")


def record_prediction_metrics(summary: dict) -> None:
    _write_json(LATEST_PREDICTION_PATH, summary)
    _append_event("prediction", summary)


def record_truth_metrics(summary: dict) -> None:
    _write_json(LATEST_TRUTH_PATH, summary)
    _append_event("truth", summary)


def record_retrain_metrics(summary: dict) -> None:
    _write_json(LATEST_RETRAIN_PATH, summary)
    _append_event("retrain", summary)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_event_counts() -> Counter:
    counts = Counter()

    if not EVENT_LOG_PATH.exists():
        return counts

    for line in EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        event = json.loads(line)
        event_type = event.get("event_type", "unknown")
        status = event.get("status", "completed")
        counts[(event_type, status)] += 1

    return counts


def _set_gauge(gauge: Gauge, value: Any, labels: dict | None = None) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        value = 1 if value else 0

    labels = labels or {}
    if labels:
        gauge.labels(**labels).set(float(value))
    else:
        gauge.set(float(value))


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render_metrics() -> tuple[bytes, str]:
    registry = CollectorRegistry()

    prediction_batches = PromCounter(
        "lms_prediction_batches",
        "Total prediction batches processed.",
        ["status"],
        registry=registry,
    )
    truth_batches = PromCounter(
        "lms_truth_batches",
        "Total truth batches evaluated.",
        ["status"],
        registry=registry,
    )
    retrain_runs = PromCounter(
        "lms_retrain_runs",
        "Total retrain flow runs.",
        ["status"],
        registry=registry,
    )

    batch_rows = Gauge(
        "lms_batch_rows",
        "Latest batch row counts by flow and metric type.",
        ["flow", "metric_type"],
        registry=registry,
    )
    data_drift_share = Gauge(
        "lms_data_drift_share",
        "Latest data drift share.",
        registry=registry,
    )
    drifted_feature_count = Gauge(
        "lms_drifted_feature_count",
        "Latest number of drifted features.",
        registry=registry,
    )
    total_feature_count = Gauge(
        "lms_total_feature_count",
        "Total feature count used for drift checks.",
        registry=registry,
    )
    prediction_score_mean = Gauge(
        "lms_prediction_score_mean",
        "Latest prediction risk score mean.",
        registry=registry,
    )
    prediction_score_drift = Gauge(
        "lms_prediction_score_drift",
        "Latest absolute prediction score mean drift.",
        registry=registry,
    )
    data_drift_detected = Gauge(
        "lms_data_drift_detected",
        "Whether latest prediction batch has data drift.",
        registry=registry,
    )
    classification_f1 = Gauge(
        "lms_classification_f1_risk",
        "Latest risk-class F1.",
        registry=registry,
    )
    classification_recall = Gauge(
        "lms_classification_recall_risk",
        "Latest risk-class recall.",
        registry=registry,
    )
    classification_precision = Gauge(
        "lms_classification_precision_risk",
        "Latest risk-class precision.",
        registry=registry,
    )
    false_negative_count = Gauge(
        "lms_false_negative_count",
        "Latest false negative count.",
        registry=registry,
    )
    performance_drift_detected = Gauge(
        "lms_performance_drift_detected",
        "Whether latest truth batch has performance drift.",
        registry=registry,
    )
    retrain_decision = Gauge(
        "lms_retrain_decision",
        "Latest retrain decision.",
        registry=registry,
    )
    promotion_decision = Gauge(
        "lms_promotion_decision",
        "Latest promotion decision.",
        registry=registry,
    )
    champion_version = Gauge(
        "lms_model_champion_version",
        "Latest champion model version observed by prediction flow.",
        ["model_name", "model_alias"],
        registry=registry,
    )

    for (event_type, status), count in _read_event_counts().items():
        if event_type == "prediction":
            prediction_batches.labels(status=status).inc(count)
        elif event_type == "truth":
            truth_batches.labels(status=status).inc(count)
        elif event_type == "retrain":
            retrain_runs.labels(status=status).inc(count)

    prediction = _read_json(LATEST_PREDICTION_PATH)
    truth = _read_json(LATEST_TRUTH_PATH)
    retrain = _read_json(LATEST_RETRAIN_PATH)

    if prediction:
        drift_metrics = prediction.get("drift_metrics", {})

        _set_gauge(batch_rows, prediction.get("row_count"), {"flow": "prediction", "metric_type": "predicted"})
        _set_gauge(data_drift_share, drift_metrics.get("lms_data_drift_share"))
        _set_gauge(drifted_feature_count, drift_metrics.get("lms_drifted_feature_count"))
        _set_gauge(total_feature_count, drift_metrics.get("lms_total_feature_count"))
        _set_gauge(prediction_score_mean, drift_metrics.get("lms_prediction_score_mean"))
        _set_gauge(prediction_score_drift, drift_metrics.get("lms_prediction_score_drift"))
        _set_gauge(data_drift_detected, drift_metrics.get("data_drift_detected"))

        version = _safe_int(prediction.get("model_version"))
        if version is not None:
            champion_version.labels(
                model_name=str(prediction.get("model_name", "unknown")),
                model_alias="champion",
            ).set(version)

    if truth:
        metrics = truth.get("metrics", {})
        performance_drift = truth.get("performance_drift", {})

        _set_gauge(batch_rows, truth.get("truth_rows"), {"flow": "truth", "metric_type": "truth"})
        _set_gauge(batch_rows, truth.get("matched_rows"), {"flow": "truth", "metric_type": "matched"})
        _set_gauge(classification_f1, metrics.get("f1_risk"))
        _set_gauge(classification_recall, metrics.get("recall_risk"))
        _set_gauge(classification_precision, metrics.get("precision_risk"))
        _set_gauge(false_negative_count, metrics.get("false_negative_count"))
        _set_gauge(performance_drift_detected, performance_drift.get("performance_drift_detected"))
        _set_gauge(retrain_decision, truth.get("retrain_decision"))

    if retrain:
        _set_gauge(promotion_decision, retrain.get("promotion_decision"))

    return generate_latest(registry), CONTENT_TYPE_LATEST
