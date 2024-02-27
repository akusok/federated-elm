import argparse

from basic_elm import ELM
from sklearn.metrics import log_loss

import flwr as fl
import utils
from flwr_datasets import FederatedDataset


if __name__ == "__main__":
    N_CLIENTS = 10
    N_ROUNDS = 20

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.node_id

    # Load the partition data
    fds = FederatedDataset(dataset="mnist", partitioners={"train": N_CLIENTS})

    dataset = fds.load_partition(partition_id, "train").with_format("numpy")
    X, y = dataset["image"].reshape((len(dataset), -1)), dataset["label"]
    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
    
    # subsample for faster test
    subsample = 3
    X_test, y_test = X_test[::subsample], y_test[::subsample]

    # part of data to share at consecutive rounds
    samples_per_round = X_train.shape[0] // N_ROUNDS

    # Create LogisticRegression Model
    model = ELM()

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)

            server_round = config["server_round"]
            n_train_a = samples_per_round * (server_round + 1)
            n_train_b = n_train_a + samples_per_round
            if n_train_a < X_train.shape[0]:  # have more data to train on
                model.fit(
                    X_train[n_train_a:n_train_b],
                    y_train[n_train_a:n_train_b]
                )

            print(f"Training finished for round {server_round}")
            return utils.get_model_parameters(model), model.get_n_training_samples(), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:8085", client=MnistClient().to_client()
    )
