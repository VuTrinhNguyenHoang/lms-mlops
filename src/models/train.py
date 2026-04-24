from data.loading import load_csv
from data.validation import validate_truth_df, get_feature_target
from data.splitting import split_train_valid
from models.configs import MODELS
from models.factory import build_pipeline
from models.evaluate import evaluate_binary_classifier
from models.registry import (
    log_candidate_run,
    register_model_version,
    set_champion_alias
)
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

def train_and_register_champion(csv_path: str):
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

        mlflow_info = log_candidate_run(
            model_name=model_name,
            model=model,
            metrics=metrics,
            params=MODELS[model_name]["params"],
            X_sample=X_valid,
        )

        results[model_name] = {
            "model": model,
            "metrics": metrics,
            "mlflow": mlflow_info,
        }
    
    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"][PRIMARY_METRIC],
    )

    best_model_uri = results[best_model_name]["mlflow"]["model_uri"]
    model_version = register_model_version(best_model_uri)
    set_champion_alias(model_version.version)

    return {
        "best_model_name": best_model_name,
        "registered_model_name": model_version.name,
        "model_version": model_version.version,
        "champion_alias": "champion",
        "metrics": results[best_model_name]["metrics"],
        "results": results,
    }
