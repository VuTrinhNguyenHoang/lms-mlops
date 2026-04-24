RANDOM_STATE = 42
TEST_SIZE = 0.2

RISK_THRESHOLD = 0.5

RISK_LEVEL_LOW = 0.3
RISK_LEVEL_HIGH = 0.7

PRIMARY_METRIC = "f1_risk"

def risk_level(score: float) -> str:
    if score >= RISK_LEVEL_HIGH:
        return "high"
    if score >= RISK_LEVEL_LOW:
        return "medium"
    return "low"