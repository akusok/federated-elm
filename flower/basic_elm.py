"""
This file contains the implementation of the Extreme Learning Machine (ELM) algorithm.

The ELM algorithm is a learning algorithm for single-layer feedforward neural networks. 
It randomly assigns the input weights and biases of the network and then analytically 
determines the output weights using the least squares solver on the hidden layer 
output matrix.

Classes:
    ELM: This class represents a single-layer feedforward neural network 
    trained using the ELM algorithm. It includes methods for initializing the model, 
    fitting the model to data, predicting probabilities and classes, and scoring the model.

Functions:
    softmax: This function converts scores to probabilities.
    norm: This function normalizes input data.
"""

import numpy as np


def softmax(x):
    """Change scores to probabilities, handling all zeros case.
    """
    return np.exp(x) / (np.sum(np.exp(x), axis=1, keepdims=True) + 1e-10)

def norm(x):
    return (x - 33.0) / 75.0


class ELM:
    """
    Extreme Learning Machine (ELM) is a single-hidden layer feedforward neural network.
    It randomly assigns weights and biases to hidden layer neurons and solves the output 
    weights analytically.
    """
    def __init__(self, n_features: int = 768, n_hidden: int = 100, n_classes: int = 10) -> None:
        """
        Initialize the ELM model.

        Parameters:
        - n_hidden (int): Number of neurons in the hidden layer.
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
        # define class variables
        self.W = None
        self.b = None
        self.HH = None
        self.HY = None
        self._HH = None
        self._HY = None
        self._n_samples = None

    def init_params(self):
        """Dummy method to initialize parameters.
        """
        self.W = np.zeros((self.n_features, self.n_hidden))
        self.b = np.zeros((1, self.n_hidden))
        self._HH = np.eye(self.n_hidden)
        self._HY = np.zeros((self.n_hidden, self.n_classes))
        self._n_samples = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update inner ELM data representation.
        """
        if self.W is None:
            raise RuntimeError("You need to set the parameters first")

        if len(y.shape) == 1:
            # Convert to one-hot encoding
            y_new = np.zeros((y.shape[0], 10))
            y_new[np.arange(y.shape[0]), y] = 1
            y = y_new

        print("fit shapes", X.shape, y.shape)

        # incremental update
        H = np.tanh(norm(X) @ self.W + self.b)
        self._HH += H.T @ H
        self._HY += H.T @ y
        self._n_samples += X.shape[0]

    def get_local_HH(self):
        return self._HH
    
    def get_local_HY(self):
        return self._HY

    def get_n_training_samples(self):
        return self._n_samples

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict output probabilities that sum up to 1.
        """
        if self.HH is None:
            raise RuntimeError("You need to fit the model first")

        print("predict proba shapes", X.shape, self.W.shape, self.b.shape, self.HH.shape, self.HY.shape)

        beta = np.linalg.solve(self.HH, self.HY)

        H = np.tanh(norm(X) @ self.W + self.b)
        yh = H @ beta

        print("predict proba shapes", X.shape, yh.shape)

        return softmax(yh)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output classes for the input data.
        """
        return self.predict_proba(X).argmax(axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        return (y_pred == y).mean()
    