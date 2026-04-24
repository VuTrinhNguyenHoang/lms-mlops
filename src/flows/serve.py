from prefect import serve

from flows.predict_flow import predict_batch_flow
from flows.retrain_flow import retrain_flow
from flows.train_flow import train_initial_champion_flow
from flows.truth_flow import evaluate_truth_flow


if __name__ == "__main__":
    serve(
        train_initial_champion_flow.to_deployment(name="train-initial-champion"),
        predict_batch_flow.to_deployment(name="predict-batch"),
        evaluate_truth_flow.to_deployment(name="evaluate-truth"),
        retrain_flow.to_deployment(name="retrain-model"),
    )
