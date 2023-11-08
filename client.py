import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class Client:
    def __init__(self, client_index, scaler, records):
        """Records are dataset records.
        """
        if client_index not in (1,2,3):
            raise ValueError("Clients support only numbers 1,2,3")

        X = records[client_index]["X"]
        Y = records[client_index]["Y"]
        self.n = records[client_index]["n_train"]
        
        # data is private, nobody can read from the outside
        X = np.array(scaler.transform(X))
        Y = np.array(Y)
        self.__X, self.__X_test, self.__Y, self.__Y_test = train_test_split(X, Y, train_size=self.n)

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
    def __H(self):
        self._has_elm()
        XW = self.__X @ self.W + self.bias
        return np.tanh(XW)
    
    @property
    def HH(self):
        H = self.__H
        return H.T@H

    @property
    def HY(self):
        H = self.__H
        return H.T@self.__Y

    @property
    def B(self):
        return self._B
    
    @B.setter
    def B(self, value):
        self._B = value

    @property
    def r2_train(self):
        self._has_elm()
        Yh = self.__H @ self.B
        return r2_score(y_true=self.__Y, y_pred=Yh)
        
    @property
    def r2(self):
        self._has_elm()
        if self.B is None:
            raise ValueError("Must set B first")
        XsW = self.__X_test @ self.W + self.bias
        Yh = np.tanh(XsW) @ self.B
        return r2_score(y_true=self.__Y_test, y_pred=Yh)

    def batch_data(self, bsize=100):
        print(f"Return {len(range(0, self.n, bsize))} batches of size {bsize}")
        H = self.__H
        Y = self.__Y
        for i in range(0, self.n, bsize):
            bH = H[i: i+bsize]
            bY = Y[i: i+bsize]
            # normalize matrices to 1-sample
            # count = min(self.n - i, bsize)  
            count = 1
            yield (bH.T@bH / count, bH.T@bY / count)

    def raw_batch_data(self, bsize=100):
        """Test for limits of federated learning with arbitrary models
        """
        X = self.__X
        Y = self.__Y
        for i in range(0, self.n, bsize):
            bX = X[i: i+bsize]
            bY = Y[i: i+bsize]
            yield bX, bY

    def raw_r2(self, model):
        return r2_score(y_true=self.__Y_test, y_pred=model.predict(self.__X_test))


class ClientNoiseHH(Client):
    def __init__(self, client_index, scaler, records):
        super().__init__(client_index, scaler, records)
        self._noise_HH = None
        self._noise_HH_matrix = None
    
    @property
    def noise_HH(self):
        return self._noise_HH
    
    @noise_HH.setter
    def noise_HH(self, value):
        self._has_elm()
        self._noise_HH = value
        A_noise = np.random.randn(self.L, self.L)
        self._noise_HH_matrix = value * (A_noise.T @ A_noise) / self.L**0.5

    @property
    def HH(self):
        H = self.__H
        return H.T@H + self.noise_HH_matrix

    def batch_data(self, bsize=100):
        gen = super().batch_data(bsize)
        for hh, hy in gen:
            yield (hh + self._noise_HH_matrix, hy)
    