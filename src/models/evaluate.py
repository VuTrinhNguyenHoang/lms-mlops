from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.contracts import TARGET_COLUMN

def _has_two_classes(y_true) -> bool:
    return len(set(y_true)) == 2

def evaluate_binary_predictions(y_true, y_pred, y_score) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_risk": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_risk": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_risk": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_negative_count": int(((y_true == 1) & (y_pred == 0)).sum()),
    }

    if _has_two_classes(y_true):
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["log_loss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None

    return metrics

def evaluate_binary_classifier(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return evaluate_binary_predictions(y, pred, proba)

def evaluate_prediction_output(
    joined_df,
    target_column: str = TARGET_COLUMN,
    label_column: str = "predicted_label",
    score_column: str = "risk_score",
) -> dict:
    y_true = joined_df[target_column]
    y_pred = joined_df[label_column]
    y_score = joined_df[score_column]

    return evaluate_binary_predictions(y_true, y_pred, y_score)
