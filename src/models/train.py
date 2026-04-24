from data.loading import load_csv
from data.validation import validate_truth_df, get_feature_target
from data.splitting import split_train_valid
from models.configs import MODELS
from models.factory import build_pipeline
from models.evaluate import evaluate_binary_classifier
from core.config import PRIMARY_METRIC, RISK_THRESHOLD

def train_candidates(csv_path: str):
    df = validate_truth_df(load_csv(csv_path))
    X, y = get_feature_target(df)

    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    results = {}

    for model_name in MODELS:
        model = build_pipeline(model_name)
        model.fit(X_train, y_train)

        metrics = evaluate_binary_classifier(
            model,
            X_valid,
            y_valid,
            threshold=RISK_THRESHOLD,
        )

        results[model_name] = {
            "model": model,
            "metrics": metrics,
        }

    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"][PRIMARY_METRIC],
    )

    return {
        "best_model_name": best_model_name,
        "best_model": results[best_model_name]["model"],
        "results": results,
    }
