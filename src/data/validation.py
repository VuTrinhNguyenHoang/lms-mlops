from core.contracts import (
    ID_COLUMNS,
    FEATURE_COLUMNS,
    PREDICTION_OUTPUT_COLUMNS,
    TARGET_COLUMN,
    PREDICTION_REQUIRED_COLUMNS,
    TRUTH_REQUIRED_COLUMNS
)

def _missing_columns(df, required_columns):
    return [col for col in required_columns if col not in df.columns]

def validate_prediction_df(df):
    missing = _missing_columns(df, PREDICTION_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {missing}")

    return df[PREDICTION_REQUIRED_COLUMNS].copy()

def validate_truth_df(df):
    missing = _missing_columns(df, TRUTH_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Truth CSV missing columns: {missing}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN} contains missing values")

    invalid_targets = set(df[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_targets:
        raise ValueError(f"Target must be binary 0/1. Invalid values: {invalid_targets}")

    return df[TRUTH_REQUIRED_COLUMNS].copy()

def validate_prediction_output_df(df):
    missing = _missing_columns(df, PREDICTION_OUTPUT_COLUMNS)
    if missing:
        raise ValueError(f"Prediction output missing columns: {missing}")

    if df["risk_score"].isna().any():
        raise ValueError("Prediction output contains missing risk_score")

    invalid_scores = df[(df["risk_score"] < 0) | (df["risk_score"] > 1)]
    if not invalid_scores.empty:
        raise ValueError("risk_score must be between 0 and 1")

    invalid_labels = set(df["predicted_label"].dropna().unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"predicted_label must be binary 0/1. Invalid values: {invalid_labels}")

    return df[PREDICTION_OUTPUT_COLUMNS].copy()

def get_feature_target(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
