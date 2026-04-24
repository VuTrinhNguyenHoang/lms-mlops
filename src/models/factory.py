from sklearn.pipeline import Pipeline
from features.builder import LMSFeatureBuilder
from models.configs import MODELS

def build_pipeline(model_name: str):
    config = MODELS[model_name]
    estimator_cls = config["estimator"]
    estimator = estimator_cls(**config["params"])

    return Pipeline(
        steps=[
            ("feature_builder", LMSFeatureBuilder()),
            ("model", estimator),
        ]
    )
