# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% editable=true slideshow={"slide_type": ""}
import numpy as np
import pandas as pd
import pygwalker as pig
import seaborn as sn
from matplotlib import pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Prepare data

# %%
# check this code complexity
# !radon mi client.py
# !radon cc client.py -a

# %%
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = data.target

# %%
# Split in geographical blocks

data_records = {
    1: {"n_train": 100, "X": X[X.Latitude > 40], "Y": Y[X.Latitude > 40]},
    2: {"n_train": 1000, "X": X[X.Longitude > -118], "Y": Y[X.Longitude > -118]},
    3: {"n_train": 10_000, "X": X[(X.Longitude <= -118) & (X.Latitude <= 40)], "Y": Y[(X.Longitude <= -118) & (X.Latitude <= 40)]},
}

# %%
# split randomly

X1, X23, Y1, Y23 = train_test_split(X, Y, train_size = data_records[1]["X"].shape[0])
X2, X3, Y2, Y3 = train_test_split(X23, Y23, train_size = data_records[2]["X"].shape[0])

data_records_random = {
    1: {"n_train": 100, "X": X1, "Y": Y1},
    2: {"n_train": 1000, "X": X2, "Y": Y2},
    3: {"n_train": 10_000, "X": X3, "Y": Y3},
}

# %%
X.shape

# %%
pig.walk(X.iloc[::4])


# %% [markdown]
# # Create clients

# %%
class client:
    def __init__(self, client_index, scaler):
        if client_index not in (1,2,3):
            raise ValueError("Clients support only numbers 1,2,3")

        records = data_records
        # records = data_records_random
        
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


# %%
scaler = RobustScaler().fit(X)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Get clients

# %%
# create ELM

n = X.shape[1]  # 8 inputs
L = 50
W = np.random.randn(n, L)
bias = np.random.randn(1, L)

c1 = client(1, scaler)
c2 = client(2, scaler)
c3 = client(3, scaler)

c1.init_elm(L, W, bias)
c2.init_elm(L, W, bias)
c3.init_elm(L, W, bias)


# %% [markdown]
# ## Find independent performance in 100-batches

# %%
def get_optimal_performance(client, batch_size=100):
    L2 = np.logspace(-5, 3, num=20)
    HH = 0
    HY = 0
    
    r2_test = []
    
    for hh, hy in client.batch_data(batch_size):
        HH += hh
        HY += hy
        
        # find best performance
        r2 = -999
        for l2 in L2:
            client.B = np.linalg.lstsq(HH + l2*np.eye(L), HY, rcond=None)[0]
            r2 = max(r2, client.r2)
        r2_test.append(r2)
        print(".", end="")

    print()
    return r2_test


# %%
r2_c1 = get_optimal_performance(c1, 5)
r2_c2 = get_optimal_performance(c2, 50)
r2_c3 = get_optimal_performance(c3, 200)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2)
plt.plot(np.arange(1, len(r2_c3)+1)*200, r2_c3)

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-1, 1])
plt.xscale("log")
plt.show()


# %% [markdown]
# ## Extend single client performance with more data

# %%
def get_optimal_performance_extended(client, collab_client, batch_size=100):
    L2 = np.logspace(-5, 3, num=30)
    HH = client.HH
    HY = client.HY    
    r2_extended = []

    r2 = -999
    for l2 in L2:
        client.B = np.linalg.lstsq(HH + l2*np.eye(L), HY, rcond=None)[0]
        r2 = max(r2, client.r2)
    r2_extended.append(r2)
    
    # streaming data from collab client, evaluate for the original client
    for hh, hy in collab_client.batch_data(batch_size):
        HH += hh
        HY += hy
        
        # find best performance
        r2 = -999
        for l2 in L2:
            client.B = np.linalg.lstsq(HH + l2*np.eye(L), HY, rcond=None)[0]
            r2 = max(r2, client.r2)
        r2_extended.append(r2)
        print(".", end="")

    print()
    return r2_extended


# %%
r2_c1_c2 = get_optimal_performance_extended(c1, c2, 50)
r2_c1_c3 = get_optimal_performance_extended(c1, c3, 200)
r2_c2_c3 = get_optimal_performance_extended(c2, c3, 200)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1, '-b')
plt.plot(np.arange(0, len(r2_c1_c2))*50 + 100, r2_c1_c2, '--b')
plt.plot(np.arange(0, len(r2_c1_c3))*200 + 100, r2_c1_c3, '--b')

plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2, c="orange")
plt.plot(np.arange(0, len(r2_c2_c3))*200 + 1000, r2_c2_c3, '--', c="orange")

plt.plot(np.arange(1, len(r2_c3)+1)*200, r2_c3)

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-1, 1])
plt.xscale("log")
plt.show()

# %% [markdown]
# ## Test with raw data and custom model

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
model = RandomForestRegressor(n_jobs=4)


# %%
def raw_get_performance(client, batch_size=10):
    X = np.empty([0, 8])
    Y = np.empty([0, ])
    r2_test = []
    
    for bx, by in client.raw_batch_data(batch_size):
        X = np.vstack([X, bx])
        Y = np.hstack([Y, by])
        model.fit(X, Y)
        
        r2_test.append(client.raw_r2(model))
        print(".", end="")
    
    print()
    return r2_test


# %%
r2_c1 = raw_get_performance(c1, 5)
r2_c2 = raw_get_performance(c2, 50)
r2_c3 = raw_get_performance(c3, 200)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2)
plt.plot(np.arange(1, len(r2_c3)+1)*200, r2_c3)

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-1, 1])
plt.xscale("log")
plt.show()


# %% [markdown]
# ## Extend data, custom model

# %%
def raw_get_performance_extended(client, collab_client, batch_size=100):
    X, Y = next(client.raw_batch_data(bsize=1_000_000))
    r2_extended = []
    
    for bx, by in collab_client.raw_batch_data(batch_size):
        X = np.vstack([X, bx])
        Y = np.hstack([Y, by])
        model.fit(X, Y)
        
        r2_extended.append(client.raw_r2(model))
        print(".", end="")
    
    print()
    return r2_extended


# %%
r2_c1_c2 = raw_get_performance_extended(c1, c2, 50)
r2_c1_c3 = raw_get_performance_extended(c1, c3, 200)
r2_c2_c3 = raw_get_performance_extended(c2, c3, 200)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2)
plt.plot(np.arange(1, len(r2_c3)+1)*200, r2_c3)

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-1, 1])
plt.xscale("log")
plt.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Run stuff

# %% editable=true slideshow={"slide_type": ""}
c1 = client(1, scaler)

# %%
# create ELM

n = X.shape[1]  # 8 inputs
L = 999
W = np.random.randn(n, L)
bias = np.random.randn(1, L)

# %%
c1.init_elm(L, W, bias)

# %%
c1.HH[:2, :4], c1.HY[:4]

# %%
B = np.linalg.lstsq(c1.HH + 1e+1*np.eye(L), c1.HY, rcond=None)[0]
c1.B = B
print(c1.r2_train)
print(c1.r2)

# %%
a = np.logspace(-5, 3)
r2_train = []
r2_test = []

for a1 in a:
    B = np.linalg.lstsq(c1.HH + a1*np.eye(L), c1.HY, rcond=None)[0]
    c1.B = B
    r2_train.append(c1.r2_train)
    r2_test.append(c1.r2)

# %%
sn.lineplot(x=np.log10(a), y=r2_train)
sn.lineplot(x=np.log10(a), y=r2_test)
sn.lineplot(x=[-5, 3], y=[0, 0])
plt.ylim(-1, 1.1)

# %%
sn.lineplot(x=np.log10(a), y=r2_train)
sn.lineplot(x=np.log10(a), y=r2_test)
sn.lineplot(x=[-5, 3], y=[0, 0])
plt.ylim(-0.1, 0.05)

# %%
