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
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from client import ClientNoiseHH

# %%
# %config InlineBackend.figure_format='retina'
plt.rcParams['figure.figsize'] = [10, 6]
np.set_printoptions(precision=3)

# %% [markdown]
# ## Prepare data

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
print(X.shape)

# %% [markdown]
# ## Get clients

# %%
# records = data_records
records = data_records_random

# %%
scaler = RobustScaler().fit(X)

# %%
# create ELM

n = X.shape[1]  # 8 inputs
L = 99
W = np.random.randn(n, L)
bias = np.random.randn(1, L)

c1 = ClientNoiseHH(1, scaler, records)
c2 = ClientNoiseHH(2, scaler, records)
c3 = ClientNoiseHH(3, scaler, records)

c1.init_elm(L, W, bias)
c2.init_elm(L, W, bias)
c3.init_elm(L, W, bias)

# %% [markdown]
# ## Look at HH values with "acceptable" noise of 10**-1

# %%
print("no noise")
c1.noise_H = 0
print(c1.HH[:3,:5])

print("noise of 10**-1")
c1.noise_H = 0.1
print(c1.HH[:3,:5])

# %% [markdown]
# ## Try to reconstruct inputs

# %%
c1.noise_H = 0.1

X = c1.X
H = c1.H
H_clean = np.tanh(X@W + bias)

# %%
print(X.mean(), X.std())


# %%
def recover(H, lim=1e5):
    XWr = np.nan_to_num(np.arctanh(H), 0)
    Xr = (XWr - bias) @ np.linalg.pinv(W)
    Xr[Xr > lim] = 0
    Xr[Xr < -lim] = 0
    return Xr #[(Xr < 1e5) & (Xr > -1e5) & (~np.isnan(Xr))]


# %%
Xr_clean = recover(H_clean)

print("mean", X.mean(), " + ", Xr_clean.mean())
print("std", X.std(), " + ", Xr_clean.std())
print("diff mean", (X - Xr_clean).mean(), "  std", (X - Xr_clean).std())

# %%
Xr = recover(H)

print("mean", X.mean(), " + ", Xr.mean())
print("std", X.std(), " + ", Xr.std())
print("diff mean", (X - Xr).mean(), "  std", (X - Xr).std())


# %% [markdown]
# ## Compute share of correctly reconstructed values

# %%
def precision_curve(Xr):
    vals = np.logspace(-3, 1, num=50)
    diff = np.abs(X - Xr)
    counts = [np.sum(diff > v) for v in vals]
    n = Xr.shape[0] * Xr.shape[1]
    counts = [c/n for c in counts]
    return vals, counts


# %%
vals, counts = precision_curve(Xr_clean)
plt.plot(vals, counts)

vals, counts = precision_curve(Xr)
plt.plot(vals, counts)

plt.xscale("log")
plt.grid()
plt.ylim(0, 1.05)
plt.show()

# %% [markdown]
# ## Larger plots

# %%
data = []
vals = np.logspace(-3, 1, num=50)
X = c1.X
n = X.shape[0] * X.shape[1]
X_std = X.std()

# for noise in (-99, *np.linspace(-2, 1, num=3*8+1)):
for noise in (-99, -1.0, *np.linspace(-5, 1, num=20)):    
    c1.noise_H = 0 if noise==-99 else 10**noise
    Xr = recover(c1.H)
    diff = np.abs(X - Xr)
    for v in vals:
        p = np.sum(diff > v) / n
        data.append({"noise": noise, "val": v/X_std, "percent": p})

test_df = pd.DataFrame(data)

# %%
sns.lineplot(
    test_df, 
    x="val", y="percent", units="noise", 
    color=".7", linewidth=1, estimator=None
)

# find worse performance
sns.lineplot(
    test_df[test_df.noise > -0.9], 
    x="val", y="percent", units="noise", 
    color="r", linewidth=1, estimator=None
)

sns.lineplot(
    test_df[test_df.noise == -1.0], 
    x="val", y="percent", units="noise", 
    color="black", linestyle='dashed', linewidth=1.5, estimator=None
)

# no noise case
sns.lineplot(
    test_df[test_df.noise == -99], 
    x="val", y="percent", linewidth=2.5
)
    
plt.xlim([-0.05, 1.5])
plt.grid("minor")
plt.xlabel("difference vs original data, in STD")
plt.ylabel("different values in reconstruction")
plt.show()

# %%
