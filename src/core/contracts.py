ID_COLUMNS = ["id"]
TARGET_COLUMN = "nograd"

FEATURE_COLUMNS = [
    "absrate1", "nsusp1", "mobility1", "q1mpa1", "q2mpa1", "q3mpa1", "q4mpa1",
    "absrate2", "nsusp2", "mobility2", "q1mpa2", "q2mpa2", "q3mpa2", "q4mpa2",
    "absrate3", "nsusp3", "mobility3", "q1mpa3", "q2mpa3", "q3mpa3", "q4mpa3",
    "absrate4", "nsusp4", "mobility4", "q1mpa4", "q2mpa4", "q3mpa4", "q4mpa4",
    "absrate5", "nsusp5", "mobility5", "q1mpa5", "q2mpa5", "q3mpa5", "q4mpa5",
    "absrate6", "nsusp6", "mobility6", "q1mpa6", "q2mpa6", "q3mpa6", "q4mpa6",
    "absrate7", "nsusp7", "mobility7", "q1mpa7", "q2mpa7", "q3mpa7", "q4mpa7"
]

PREDICTION_REQUIRED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS
TRUTH_REQUIRED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
PREDICTION_OUTPUT_COLUMNS = [
    *ID_COLUMNS,
    "batch_id",
    "risk_score",
    "predicted_label",
    "risk_level",
    "model_name",
    "model_version"
]
