from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MODELS = {
    "logistic_regression": {
        "estimator": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "scale": True,
    },
    "random_forest": {
        "estimator": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 42,
        },
    },
}
