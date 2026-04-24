DEFAULT_PROMOTION_RULES = {
    "primary_metric": "f1_risk",
    "min_primary_improvement": 0.01,
    "guardrail_metrics": ["recall_risk", "precision_risk"],
    "max_guardrail_drop": 0.05,
}

def should_promote(
    candidate_metrics: dict,
    current_metrics: dict,
    rules: dict | None = None,
) -> tuple[bool, list[str]]:
    rules = rules or DEFAULT_PROMOTION_RULES
    reasons = []

    primary_metric = rules["primary_metric"]
    min_improvement = rules["min_primary_improvement"]

    candidate_primary = candidate_metrics.get(primary_metric)
    current_primary = current_metrics.get(primary_metric)

    if candidate_primary is None or current_primary is None:
        return False, [f"Missing primary metric: {primary_metric}"]

    improvement = candidate_primary - current_primary

    if improvement < min_improvement:
        return False, [
            f"{primary_metric}_improvement={improvement:.3f} < min_primary_improvement={min_improvement}"
        ]

    reasons.append(
        f"{primary_metric}_improvement={improvement:.3f} >= min_primary_improvement={min_improvement}"
    )

    for metric_name in rules["guardrail_metrics"]:
        candidate_value = candidate_metrics.get(metric_name)
        current_value = current_metrics.get(metric_name)

        if candidate_value is None or current_value is None:
            return False, [*reasons, f"Missing guardrail metric: {metric_name}"]

        drop = current_value - candidate_value

        if drop > rules["max_guardrail_drop"]:
            return False, [
                *reasons,
                f"{metric_name}_drop={drop:.3f} > max_guardrail_drop={rules['max_guardrail_drop']}",
            ]

    return True, reasons
