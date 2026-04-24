from core.config import RISK_THRESHOLD, risk_level
from core.contracts import ID_COLUMNS, FEATURE_COLUMNS
from data.loading import load_csv
from data.validation import validate_prediction_df


def predict_batch(
    model,
    csv_path: str,
    batch_id: str,
    model_name: str,
    model_version: str = "local",
):
    df = validate_prediction_df(load_csv(csv_path))

    ids = df[ID_COLUMNS].copy()
    X = df[FEATURE_COLUMNS].copy()

    risk_scores = model.predict_proba(X)[:, 1]
    predicted_labels = (risk_scores >= RISK_THRESHOLD).astype(int)

    output = ids.copy()
    output["batch_id"] = batch_id
    output["risk_score"] = risk_scores
    output["predicted_label"] = predicted_labels
    output["risk_level"] = [risk_level(score) for score in risk_scores]
    output["model_name"] = model_name
    output["model_version"] = model_version

    return output
