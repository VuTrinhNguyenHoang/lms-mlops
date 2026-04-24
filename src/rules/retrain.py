DEFAULT_RETRAIN_RULES = {
    "min_truth_rows": 50,
    "min_matched_ratio": 0.8,
    "retrain_on_performance_drift": True,
    "retrain_on_data_drift": True,
}

def should_retrain(evaluation_summary: dict, rules: dict | None = None) -> tuple[bool, list[str]]:
    rules = rules or DEFAULT_RETRAIN_RULES
    reasons = []

    truth_rows = evaluation_summary["truth_rows"]
    matched_rows = evaluation_summary["matched_rows"]
    matched_ratio = matched_rows / truth_rows if truth_rows else 0

    if truth_rows < rules["min_truth_rows"]:
        return False, [
            f"truth_rows={truth_rows} < min_truth_rows={rules['min_truth_rows']}"
        ]

    if matched_ratio < rules["min_matched_ratio"]:
        return False, [
            f"matched_ratio={matched_ratio:.3f} < min_matched_ratio={rules['min_matched_ratio']}"
        ]

    performance_drift = evaluation_summary.get("performance_drift", {})
    if (
        rules["retrain_on_performance_drift"]
        and performance_drift.get("performance_drift_detected", False)
    ):
        reasons.extend(performance_drift.get("performance_drift_reasons", []))

    data_drift = evaluation_summary.get("data_drift", {})
    if (
        rules["retrain_on_data_drift"]
        and data_drift.get("data_drift_detected", False)
    ):
        reasons.append("data_drift_detected=True")

    return bool(reasons), reasons
