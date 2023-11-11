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


# %% [markdown]
# ## Get noise effect data

# %%
test_data = []

for noise in (-99, *np.linspace(-2, 1, num=3*8+1)):
    print(noise)
    for c, bsize, idx in zip([c1, c2, c3], [5, 50, 300], [1, 2, 3]):
        c.noise_H = 0 if noise==-99 else 10**noise  # logarithmic noise with no-noise special case
        r2_c = get_optimal_performance(c, bsize)
        counts = np.arange(1, len(r2_c)+1) * bsize
        for n,r2 in zip(counts, r2_c):
            test_data.append({"client": idx, "samples": n, "noise": noise, "r2": r2})

test_df = pd.DataFrame(test_data)

# %%
for idx in [1,2,3]:
    sns.lineplot(
        test_df[test_df.client == idx], 
        x="samples", y="r2", units="noise", 
        color=".7", linewidth=1, estimator=None
    )

    # find worse performance
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise > -0.9)], 
        x="samples", y="r2", units="noise", 
        color="r", linewidth=1, estimator=None
    )

    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == -1.0)], 
        x="samples", y="r2", units="noise", 
        color="black", linestyle='dashed', linewidth=1.5, estimator=None
    )

# no noise case
for idx in [1,2,3]:
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == -99)], 
        x="samples", y="r2", linewidth=2.5
    )
    
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xscale("log")
plt.grid("major", axis="y")
plt.show()

# %%
for idx in [1,2,3]:
    sns.lineplot(
        test_df[test_df.client == idx], 
        x="samples", y="r2", units="noise", 
        color=".7", linewidth=1, estimator=None
    )

    # find worse performance
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise > -0.7)], 
        x="samples", y="r2", units="noise", 
        color="r", linewidth=1, estimator=None
    )

    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == -1.1052631578947372)], 
        x="samples", y="r2", units="noise", 
        color="black", linestyle='dashed', linewidth=1.5, estimator=None
    )

# no noise case
for idx in [1,2,3]:
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == -99)], 
        x="samples", y="r2", linewidth=2.5
    )
    
plt.plot([0, 10000], [0, 0], '-k')
# plt.ylim([-1, 1])
plt.ylim([-0.15, 0.55])
plt.xscale("log")
plt.grid("major", axis="y")
plt.show()

# %% [markdown]
# ## Look at HH values with "acceptable" noise of 10**-1

# %%
print("no noise")
c1.noise_H = 0
print(c1.HH[:3,:5])

print("noise of 10**-1")
c1.noise_H = 0.1
print(c1.HH[:3,:5])
