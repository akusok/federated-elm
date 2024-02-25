import flwr as fl
import utils
from sklearn.metrics import log_loss
from typing import Dict

from flwr_datasets import FederatedDataset
from basic_elm import ELM


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: ELM):
    """Return an evaluation function for server-side evaluation.
    """
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    dataset = fds.load_full("test").with_format("numpy")
    X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"]

    # subsample for faster test
    subsample = 6
    X_test, y_test = X_test[::subsample], y_test[::subsample]

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = ELM()
    utils.set_initial_params(model)

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8085",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=25),
    )
