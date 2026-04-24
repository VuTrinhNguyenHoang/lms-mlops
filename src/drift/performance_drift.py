DEFAULT_PERFORMANCE_DRIFT_RULES = {
    "min_recall_risk": 0.75,
    "min_f1_risk": 0.5,
    "max_false_negative_count": 10,
}

def compute_performance_drift(
    metrics: dict,
    baseline_metrics: dict | None = None,
    rules: dict | None = None,
) -> dict:
    rules = rules or DEFAULT_PERFORMANCE_DRIFT_RULES
    reasons = []

    if metrics["recall_risk"] < rules["min_recall_risk"]:
        reasons.append(
            f"recall_risk={metrics['recall_risk']:.3f} < min_recall_risk={rules['min_recall_risk']}"
        )

    if metrics["f1_risk"] < rules["min_f1_risk"]:
        reasons.append(
            f"f1_risk={metrics['f1_risk']:.3f} < min_f1_risk={rules['min_f1_risk']}"
        )

    if metrics["false_negative_count"] > rules["max_false_negative_count"]:
        reasons.append(
            f"false_negative_count={metrics['false_negative_count']} > max_false_negative_count={rules['max_false_negative_count']}"
        )

    summary = {
        "performance_drift_detected": bool(reasons),
        "performance_drift_reasons": reasons,
    }

    if baseline_metrics is not None:
        summary["f1_risk_drop"] = baseline_metrics["f1_risk"] - metrics["f1_risk"]
        summary["recall_risk_drop"] = baseline_metrics["recall_risk"] - metrics["recall_risk"]
        summary["precision_risk_drop"] = baseline_metrics["precision_risk"] - metrics["precision_risk"]

    return summary
