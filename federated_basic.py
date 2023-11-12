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

from client import Client
from utils import RECORDS_RANDOM, SCALER, get_optimal_performance, get_optimal_performance_extended

# %%
# %config InlineBackend.figure_format='retina'

# %% [markdown]
# ## Get clients

# %%
n = SCALER.n_features_in_  # number of inputs, 8 for California housing
L = 200
W = np.random.randn(n, L)
bias = np.random.randn(1, L)

def get_clients():
    clients = [Client(idx, SCALER, RECORDS_RANDOM) for idx in [1,2,3]]
    [c.init_elm(L, W, bias) for c in clients]
    return clients

clients = get_clients()

# %%
# generate batch sizes unifromly for log scale

bsize1 = np.logspace(1, 2, num=15).astype(int)
bsize2 = np.logspace(1.5, 3, num=15).astype(int)
bsize3 = np.logspace(2, 4, num=15).astype(int)
bsize = [bsize1, bsize2, bsize3]

# %% [markdown]
# ## Basic federated performance

# %%
test_data = []

for run in range(10):
    print(run, end="  ")
    clients = get_clients()
    for idx, c in enumerate(clients):
        r2_c = get_optimal_performance(c, None, bsize[idx])
        for n, r2 in zip(bsize[idx], r2_c):
            test_data.append({"run": run, "client": idx, "samples": n, "r2": r2})

test_df = pd.DataFrame(test_data)

# %%
test_df.to_csv("data_basic.csv")

# %%
plt.rcParams['figure.figsize'] = [5, 3]

# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_df[test_df.client == idx], 
        x="samples", y="r2", linewidth=2.5, label=name
    )

plt.legend()
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xlabel("training samples")
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("fig_basic.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Extend single client performance with more data

# %%
test_data_2 = []

for run in range(10):
    print(run, end="  ")
    clients = get_clients()
    
    for idx, c in enumerate(clients):
        r2_c1_c2 = get_optimal_performance_extended(c1, c2, bsize2)
        for n, r2 in zip((0, *bsize2), r2_c1_c2):
            test_data_2.append({"run": run, "client": "c1_c2", "samples": n+100, "r2": r2})

        r2_c1_c3 = get_optimal_performance_extended(c1, c3, bsize3)
        for n, r2 in zip((0, *bsize3), r2_c1_c3):
            test_data_2.append({"run": run, "client": "c1_c3", "samples": n+100, "r2": r2})

        r2_c2_c3 = get_optimal_performance_extended(c2, c3, bsize3)
        for n, r2 in zip((0, *bsize3), r2_c2_c3):
            test_data_2.append({"run": run, "client": "c2_c3", "samples": n+1000, "r2": r2})

test_df_2 = pd.DataFrame(test_data_2)

# %%
test_df_2.to_csv("data_basic_fed.csv")

# %%
test_df = pd.read_csv("data_basic.csv")

# %%
plt.rcParams['figure.figsize'] = [5, 3]

# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_df[test_df.client == idx], 
        x="samples", y="r2", linewidth=1
    )

sns.lineplot(
    test_df_2[test_df_2.client == "c1_c2"], 
    x="samples", y="r2", linestyle="--", linewidth=2, label="client 1+2"
)

sns.lineplot(
    test_df_2[test_df_2.client == "c1_c3"], 
    x="samples", y="r2", linestyle="--", linewidth=2, label="client 1+3"
)

sns.lineplot(
    test_df_2[test_df_2.client == "c2_c3"], 
    x="samples", y="r2", linestyle="--", linewidth=2, label="client 2+3"
)


plt.legend()
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xlabel("training samples")
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("fig_basic_fed.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Test with raw data and custom model

# %%
model = RandomForestRegressor(n_estimators=30, n_jobs=6)


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
r2_c3 = raw_get_performance(c3, 300)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2)
plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3)

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

    model.fit(X, Y)
    r2_extended.append(client.raw_r2(model))
    
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
r2_c1_c3 = raw_get_performance_extended(c1, c3, 300)
r2_c2_c3 = raw_get_performance_extended(c2, c3, 300)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1, '-b')
plt.plot(np.arange(0, len(r2_c1_c2))*50 + 100, r2_c1_c2, '--b')
plt.plot(np.arange(0, len(r2_c1_c3))*300 + 100, r2_c1_c3, '--b')

plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2, c="orange")
plt.plot(np.arange(0, len(r2_c2_c3))*300 + 1000, r2_c2_c3, '--', c="orange")

plt.plot(np.arange(1, len(r2_c3)+1)*200, r2_c3)

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.2, 0.8])
plt.xscale("log")
plt.show()
