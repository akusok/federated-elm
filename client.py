import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class Client:
    def __init__(self, client_index, scaler, records):
        """Records are dataset records."""
        if client_index not in (1, 2, 3):
            raise ValueError("Clients support only numbers 1,2,3")

        X = records[client_index]["X"]
        Y = records[client_index]["Y"]
        self.n = records[client_index]["n_train"]

        # data is private, nobody can read from the outside
        X = np.array(scaler.transform(X))
        Y = np.array(Y)
        self.X, self.X_test, self.Y, self.Y_test = train_test_split(X, Y, train_size=self.n)
        self.X_test = self.X_test[:400]
        self.Y_test = self.Y_test[:400]

        # future ELM params
        self.L, self.W, self.bias, self._B = None, None, None, None

    def init_elm(self, L, W, bias):
        self.L = L
        self.W = W
        self.bias = bias

    def _has_elm(self):
        if self.L is None or self.W is None or self.bias is None:
            raise ValueError("ELM not initialized")

    @property
    def H(self):
        self._has_elm()
        XW = self.X @ self.W + self.bias
        return np.tanh(XW)

    @property
    def HH(self):
        H = self.H
        return H.T @ H

    @property
    def HY(self):
        H = self.H
        return H.T @ self.Y

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        self._B = value

    @property
    def r2_train(self):
        self._has_elm()
        Yh = self.H @ self.B
        return r2_score(y_true=self.Y, y_pred=Yh)

    @property
    def r2(self):
        self._has_elm()
        if self.B is None:
            raise ValueError("Must set B first")
        XsW = self.X_test @ self.W + self.bias
        Yh = np.tanh(XsW) @ self.B
        return r2_score(y_true=self.Y_test, y_pred=Yh)

    def batch_data(self, bsize=100, batches=None):
        # if batches is None:
        #     print(f"Return {len(range(0, self.n, bsize))} batches of size {bsize}", end=" ")
        # else:
        #     print(f"Return {len(batches)} specific batches", end=" ")
        H = self.H
        Y = self.Y

        # old mode with fixed batches
        if batches is None:
            for i in range(0, self.n, bsize):
                bH = H[i : i + bsize]
                bY = Y[i : i + bsize]
                # normalize matrices to 1-sample
                # count = min(self.n - i, bsize)
                count = 1
                yield (bH.T @ bH / count, bH.T @ bY / count)

        # new mode with dynamic size batches
        batches_plus = [*batches, self.n]  # add upper boundary for last batch
        for i, b in enumerate(batches):
            j0 = b
            j1 = batches_plus[i+1]
            bH = H[j0:j1]
            bY = Y[j0:j1]
            yield (bH.T @ bH, bH.T @ bY)


    def raw_batch_data(self, bsize=100):
        """Test for limits of federated learning with arbitrary models"""
        X = self.X
        Y = self.Y
        for i in range(0, self.n, bsize):
            bX = X[i : i + bsize]
            bY = Y[i : i + bsize]
            yield bX, bY

    def raw_r2(self, model):
        return r2_score(y_true=self.Y_test, y_pred=model.predict(self.X_test))


class ClientNoiseHH(Client):
    @property
    def noise_H(self):
        return self._noise_H_value

    @noise_H.setter
    def noise_H(self, value):
        self._has_elm()
        self._noise_H_value = value
        self._noise_H = value * np.random.randn(self.n, self.L)

    @property
    def H(self):
        self._has_elm()
        XW = self.X @ self.W + self.bias
        return np.tanh(XW) + self._noise_H


class ClientNoiseY(Client):
    def __init__(self, client_index, scaler, records):
        super().__init__(client_index, scaler, records)
        self.Y_orig = self.Y.copy()

    @property
    def noise_Y(self):
        return self._noise_Y_value

    @noise_Y.setter
    def noise_Y(self, value):
        self._has_elm()
        self._noise_Y_value = value
        m, s = self.Y_orig.mean(), self.Y_orig.std()

        # generate new Y value with random added noise
        new_Y = []
        for y in self.Y_orig:
            if np.random.rand() < value:
                new_Y.append(m + s*np.random.randn())  # fake value
            else:
                new_Y.append(y)

        self.Y = np.array(new_Y)


class ClientNoiseBoth(Client):
    def __init__(self, client_index, scaler, records):
        super().__init__(client_index, scaler, records)
        self.Y_orig = self.Y.copy()

    @property
    def noise_Y(self):
        return self._noise_Y_value

    @noise_Y.setter
    def noise_Y(self, value):
        self._has_elm()
        self._noise_Y_value = value
        m, s = self.Y_orig.mean(), self.Y_orig.std()

        # generate new Y value with random added noise
        new_Y = []
        for y in self.Y_orig:
            if np.random.rand() < value:
                new_Y.append(m + s*np.random.randn())  # fake value
            else:
                new_Y.append(y)

        self.Y = np.array(new_Y)

    @property
    def noise_H(self):
        return self._noise_H_value

    @noise_H.setter
    def noise_H(self, value):
        self._has_elm()
        self._noise_H_value = value
        self._noise_H = value * np.random.randn(self.n, self.L)

    @property
    def H(self):
        self._has_elm()
        XW = self.X @ self.W + self.bias
        return np.tanh(XW) + self._noise_H
