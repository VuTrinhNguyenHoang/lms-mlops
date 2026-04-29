import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from core.config import (
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ARTIFACT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_REGISTRY_URI,
    MLFLOW_TRACKING_URI
)

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def log_candidate_run(model_name: str, model, metrics: dict, params: dict, X_sample):
    setup_mlflow()

    input_example = X_sample.head(5)
    signature = infer_signature(input_example, model.predict_proba(input_example))
    loggable_metrics = {
        key: value
        for key, value in metrics.items()
        if value is not None
    }

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(loggable_metrics)

        mlflow.set_tags(
            {
                "task": "initial_train",
                "candidate_or_champion": "candidate",
                "predict_contract": "predict_proba",
            }
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=MLFLOW_MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=input_example,
            pyfunc_predict_fn="predict_proba",
        )

        return {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }
    
def register_model_version(model_uri: str, model_name: str = MLFLOW_REGISTERED_MODEL_NAME):
    setup_mlflow()
    return mlflow.register_model(model_uri=model_uri, name=model_name)

def set_champion_alias(model_version: str, model_name: str = MLFLOW_REGISTERED_MODEL_NAME) -> None:
    setup_mlflow()
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias=MLFLOW_CHAMPION_ALIAS,
        version=model_version,
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="validation_status",
        value="approved",
    )

def get_champion_version(model_name: str = MLFLOW_REGISTERED_MODEL_NAME):
    setup_mlflow()
    client = MlflowClient()
    return client.get_model_version_by_alias(model_name, MLFLOW_CHAMPION_ALIAS)

def load_champion_model(model_name: str = MLFLOW_REGISTERED_MODEL_NAME):
    setup_mlflow()

    # Local demo only: MLflow sklearn models use pickle/cloudpickle unless using skops.
    os.environ.setdefault("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")

    model_uri = f"models:/{model_name}@{MLFLOW_CHAMPION_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)
