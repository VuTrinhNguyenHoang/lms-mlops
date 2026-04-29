from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features.builder import LMSFeatureBuilder
from models.configs import MODELS

def build_pipeline(model_name: str):
    config = MODELS[model_name]
    estimator_cls = config["estimator"]
    estimator = estimator_cls(**config["params"])

    steps = [("feature_builder", LMSFeatureBuilder())]

    if config.get("scale", False):
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", estimator))

    return Pipeline(steps=steps)
