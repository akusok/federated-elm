import numpy as np
from basic_elm import ELM

from flwr.common import NDArrays


def get_model_parameters(model: ELM) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    params = [
        model.get_local_HH(),
        model.get_local_HY(),
    ]
    return params


def set_model_params(model: ELM, params: NDArrays) -> ELM:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.HH = params[0]
    model.HY = params[1]
    return model


def set_initial_params(model: ELM):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    n_hidden = 100

    np.random.seed(42)
    model.n_features = n_features
    model.n_hidden = n_hidden
    model.n_classes = n_classes
    model.W = np.random.randn(n_features, n_hidden)
    model.b = np.random.randn(1, n_hidden)
    model._HH = np.eye(n_hidden)
    model._HY = np.zeros((n_hidden, n_classes))
