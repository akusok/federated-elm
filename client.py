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
# # Play with stuff

# %%
# check this code complexity
# !radon mi client.py
# !radon cc client.py -a

# %%
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = data.target

data_records = {
    1: {"n_train": 100, "X": X[X.Latitude > 40], "Y": Y[X.Latitude > 40]},
    2: {"n_train": 1000, "X": X[X.Longitude > -118], "Y": Y[X.Longitude > -118]},
    3: {"n_train": 10_000, "X": X[(X.Longitude <= -118) & (X.Latitude <= 40)], "Y": Y[(X.Longitude <= -118) & (X.Latitude <= 40)]},
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
            
        X = data_records[client_index]["X"]
        Y = data_records[client_index]["Y"]
        self.n = data_records[client_index]["n_train"]
        
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
    def _H(self):
        self._has_elm()
        XW = self.__X @ self.W + self.bias
        return np.tanh(XW)
    
    @property
    def HH(self):
        H = self._H
        return H.T@H

    @property
    def HY(self):
        H = self._H
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
        Yh = self._H @ self.B
        return r2_score(y_true=self.__Y, y_pred=Yh)
        
    @property
    def r2(self):
        self._has_elm()
        if self.B is None:
            raise ValueError("Must set B first")
        XsW = self.__X_test @ self.W + self.bias
        Yh = np.tanh(XsW) @ self.B
        return r2_score(y_true=self.__Y_test, y_pred=Yh)


# %% [markdown]
# ## Run some stuff

# %%
scaler = RobustScaler().fit(X)

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
