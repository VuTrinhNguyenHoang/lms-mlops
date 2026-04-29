from prefect import flow, get_run_logger

from core.config import REFERENCE_DATA_PATH
from models.train import train_and_register_champion

def _summarize_training_result(result: dict) -> dict:
    return {
        "best_model_name": result["best_model_name"],
        "registered_model_name": result["registered_model_name"],
        "model_version": result["model_version"],
        "champion_alias": result["champion_alias"],
        "metrics": result["metrics"],
        "candidate_metrics": {
            name: item["metrics"]
            for name, item in result["results"].items()
        },
        "candidate_runs": {
            name: item["mlflow"]
            for name, item in result["results"].items()
        },
    }

@flow(name="train-initial-champion", log_prints=True)
def train_initial_champion_flow(csv_path: str) -> dict:
    logger = get_run_logger()
    logger.info("Starting initial champion training from %s", csv_path)

    result = train_and_register_champion(csv_path)
    summary = _summarize_training_result(result)

    logger.info(
        "Promoted %s version %s as %s",
        summary["registered_model_name"],
        summary["model_version"],
        summary["champion_alias"],
    )

    return summary

if __name__ == "__main__":
    print(train_initial_champion_flow(str(REFERENCE_DATA_PATH)))
