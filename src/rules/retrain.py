DEFAULT_RETRAIN_RULES = {
    "min_truth_rows": 50,
    "min_matched_ratio": 0.8,
    "min_recall_risk": 0.75,
    "min_f1_risk": 0.5,
    "max_false_negative_count": 10,
}

def should_retrain(evaluation_summary: dict, rules: dict | None = None) -> tuple[bool, list[str]]:
    rules = rules or DEFAULT_RETRAIN_RULES
    reasons = []

    metrics = evaluation_summary["metrics"]

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

    return bool(reasons), reasons
