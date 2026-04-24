from data.loading import load_csv
from data.validation import validate_truth_df, get_feature_target
from data.splitting import split_train_valid
from models.configs import MODELS
from models.factory import build_pipeline
from models.evaluate import evaluate_binary_classifier
from models.registry import (
    load_champion_model,
    log_candidate_run,
    register_model_version,
    set_champion_alias,
)
from core.config import PRIMARY_METRIC, RISK_THRESHOLD

def _prepare_train_valid(csv_path: str):
    df = validate_truth_df(load_csv(csv_path))
    X, y = get_feature_target(df)
    return split_train_valid(X, y)

def _select_best_model_name(results: dict) -> str:
    return max(
        results,
        key=lambda name: results[name]["metrics"][PRIMARY_METRIC],
    )

def _fit_and_evaluate_candidates(
    X_train,
    X_valid,
    y_train,
    y_valid,
    log_to_mlflow: bool = False,
) -> dict:
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

        result = {
            "model": model,
            "metrics": metrics,
        }

        if log_to_mlflow:
            result["mlflow"] = log_candidate_run(
                model_name=model_name,
                model=model,
                metrics=metrics,
                params=MODELS[model_name]["params"],
                X_sample=X_valid,
            )

        results[model_name] = result

    return results

def train_candidates(csv_path: str) -> dict:
    X_train, X_valid, y_train, y_valid = _prepare_train_valid(csv_path)
    results = _fit_and_evaluate_candidates(
        X_train,
        X_valid,
        y_train,
        y_valid,
        log_to_mlflow=False,
    )

    best_model_name = _select_best_model_name(results)

    return {
        "best_model_name": best_model_name,
        "best_model": results[best_model_name]["model"],
        "metrics": results[best_model_name]["metrics"],
        "results": results,
    }

def train_and_log_candidates(
    csv_path: str,
    include_champion_baseline: bool = False,
) -> dict:
    X_train, X_valid, y_train, y_valid = _prepare_train_valid(csv_path)
    results = _fit_and_evaluate_candidates(
        X_train,
        X_valid,
        y_train,
        y_valid,
        log_to_mlflow=True,
    )

    best_model_name = _select_best_model_name(results)

    output = {
        "best_model_name": best_model_name,
        "best_model": results[best_model_name]["model"],
        "metrics": results[best_model_name]["metrics"],
        "results": results,
    }

    if include_champion_baseline:
        champion = load_champion_model()
        output["champion_metrics"] = evaluate_binary_classifier(
            champion,
            X_valid,
            y_valid,
            threshold=RISK_THRESHOLD,
        )

    return output

def train_and_register_champion(csv_path: str) -> dict:
    result = train_and_log_candidates(csv_path)

    best_model_name = result["best_model_name"]
    best_model_uri = result["results"][best_model_name]["mlflow"]["model_uri"]

    model_version = register_model_version(best_model_uri)
    set_champion_alias(model_version.version)

    return {
        "best_model_name": best_model_name,
        "registered_model_name": model_version.name,
        "model_version": model_version.version,
        "champion_alias": "champion",
        "metrics": result["metrics"],
        "results": result["results"],
    }
