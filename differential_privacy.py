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
from sklearn.ensemble import RandomForestRegressor

from client import ClientNoiseY
from utils import (
    RECORDS_RANDOM, 
    RECORDS_GEOGRAPHICAL, 
    SCALER, 
    get_optimal_performance, 
    get_optimal_performance_extended
)

# %%
# %config InlineBackend.figure_format='retina'
plt.rcParams['figure.figsize'] = [4, 3]

# %% [markdown]
# ## Get clients

# %%
n = SCALER.n_features_in_  # number of inputs, 8 for California housing
L = 200
W = np.random.randn(n, L)
bias = np.random.randn(1, L)

def get_clients():
    clients = [ClientNoiseY(idx, SCALER, RECORDS_RANDOM) for idx in [1,2,3]]
    for c in clients:
        c.init_elm(L, W, bias)
        c.noise_Y = 0
    return clients

clients = get_clients()

# %%
# generate batch sizes unifromly for log scale

bsize1 = np.logspace(1, 2, num=15).astype(int)
bsize2 = np.logspace(1.5, 3, num=15).astype(int)
bsize3 = np.logspace(2, 4, num=15).astype(int)
bsize = [bsize1, bsize2, bsize3]

# %% [markdown]
# ## Get noise effect data

# %%
test_data = []

for run in range(10):
    print()
    print(run, end="  ")
    clients = get_clients()
    
    # for noise in (*range(15), *range(15, 96, 5)):
    for noise in range(0, 20, 3):
        print(".", end="")
        for idx, c in enumerate(clients):
            c.noise_Y = 0.01*noise  # noise in percent
            r2_c = get_optimal_performance(c, None, bsize[idx])
            for n, r2 in zip(bsize[idx], r2_c):
                test_data.append({
                    "run": run, "client": idx, "samples": n, "noise": noise, "r2": r2
                })

test_df = pd.DataFrame(test_data)
test_df.to_csv("data_diff_privacy.csv")

# %% [markdown]
# ## Differential privacy effect

# %%
test_df = pd.read_csv("data_diff_privacy.csv")

# %%
# test_df = test_df[(test_df.noise % 5 == 0) & (test_df.noise < 51)]

for idx in range(3):
    for noise in range(3, 20, 3):
        sns.lineplot(
            test_df[
                (test_df.client == idx)
                & (test_df.noise == noise)
            ], 
            x="samples", y="r2", 
            linewidth=1, color=".7", err_style=None
        )

# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_df[
            (test_df.client == idx)
            & (test_df.noise == 0)
        ], 
        x="samples", y="r2", linewidth=2.5, label=name
    )

    
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("noise_Y.pdf", bbox_inches="tight")
plt.show()
