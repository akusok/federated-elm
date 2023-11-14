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

from client import ClientNoiseY
from utils import (
    RECORDS_RANDOM, 
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
    
    for noise in (*range(15), *range(15, 96, 5)):
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
for idx in range(3):
    for noise in range(20, 51, 10):
        sns.lineplot(
            test_df[
                (test_df.client == idx)
                & (test_df.noise == noise)
            ], 
            x="samples", y="r2", 
            linewidth=1, color=".7", err_style=None
        )

# optimal noise case
for idx in range(3):
    sns.lineplot(
        test_df[
            (test_df.client == idx)
            & (test_df.noise == 10)
        ], 
        x="samples", y="r2", errorbar=None,
        color="black", linestyle=':', linewidth=1.5, 
        label="noise=10%" if idx==2 else None
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


# %% [markdown]
# ## Extending clean local training with noisy federated data

# %%
def noise_block(c1, c2, c3, run, noise):
    c1.noise_Y = 0
    c2.noise_Y = 0
    c3.noise_Y = noise
    temp_data = []
    
    r2_c1_c3 = get_optimal_performance_extended(c1, c3, bsize3)
    # first data point comes without extra training samples, so add 0 to bsizes array
    for n, r2 in zip((0, *bsize3), r2_c1_c3):
        temp_data.append({"run": run, "client": "c1_c3", "samples": n+100, "noise": noise, "r2": r2})

    r2_c2_c3 = get_optimal_performance_extended(c2, c3, bsize3)
    for n, r2 in zip((0, *bsize3), r2_c2_c3):
        temp_data.append({"run": run, "client": "c2_c3", "samples": n+1000, "noise": noise, "r2": r2})

    c2.noise_Y = noise
    r2_c1_c2 = get_optimal_performance_extended(c1, c2, bsize2)
    for n, r2 in zip((0, *bsize2), r2_c1_c2):
        temp_data.append({"run": run, "client": "c1_c2", "samples": n+100, "noise": noise, "r2": r2})    

    return temp_data


# %%
test_data_fed = []

for run in range(20):
    print(run, end="  ")
    clients = get_clients()
    c1, c2, c3 = clients
    
    for idx, c in enumerate(clients):
        c.noise_Y = 0
        r2_c = get_optimal_performance(c, None, bsize[idx])
        for n, r2 in zip(bsize[idx], r2_c):
            test_data_fed.append({"run": run, "client": idx, "samples": n, "noise": 0.0, "r2": r2})

    for test_noise_level in (0, 0.10, 0.20, 0.35):
        test_data_fed.extend(noise_block(c1, c2, c3, run, noise=test_noise_level))

test_fed_df = pd.DataFrame(test_data_fed)
test_fed_df.to_csv("data_diff_privacy_federated_4.csv")

# %% [markdown]
# ## Print federated learning with noise levels

# %%
test_fed_df = pd.read_csv("data_diff_privacy_federated_4.csv")

# %%
# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_fed_df[
            (test_fed_df.client == str(idx))
            & (test_fed_df.noise == 0.0)
        ], 
        x="samples", y="r2", linewidth=1, label=name
    )

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c2")
        & (test_fed_df.noise == 0.0)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+2"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c3")
        & (test_fed_df.noise == 0.0)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+3"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c2_c3")
        & (test_fed_df.noise == 0.0)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 2+3"
)

    
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("noise_Y_fed.pdf", bbox_inches="tight")
plt.show()

# %%
noise = 0.10

# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_fed_df[
            (test_fed_df.client == str(idx))
            & (test_fed_df.noise == 0.0)
        ], 
        x="samples", y="r2", linewidth=1, label=name
    )

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c2")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+2"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c3")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+3"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c2_c3")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 2+3"
)

    
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("noise_Y_fed_10p.pdf", bbox_inches="tight")
plt.show()

# %%
noise = 0.20

# no noise case
for idx, name in zip([0,1], ["client 1", "client 2"]):
    sns.lineplot(
        test_fed_df[
            (test_fed_df.client == str(idx))
            & (test_fed_df.noise == 0.0)
        ], 
        x="samples", y="r2", linewidth=1, label=name
    )

sns.lineplot(
    test_df[
        (test_df.client == 2)
        & (test_df.noise == 20)
    ], 
    x="samples", y="r2", linewidth=1, errorbar=None,  
    linestyle="-", label="client 3 at\n20% noise"
)


sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c2")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+2"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c3")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+3"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c2_c3")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 2+3"
)

    
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("noise_Y_fed_20p.pdf", bbox_inches="tight")
plt.show()

# %%
noise = 0.35

# no noise case
for idx, name in zip([0,1], ["client 1", "client 2"]):
    sns.lineplot(
        test_fed_df[
            (test_fed_df.client == str(idx))
            & (test_fed_df.noise == 0.0)
        ], 
        x="samples", y="r2", linewidth=1, label=name
    )

sns.lineplot(
    test_df[
        (test_df.client == 2)
        & (test_df.noise == 35)
    ], 
    x="samples", y="r2", linewidth=1, errorbar=None,  
    linestyle="-", label="client 3 at\n35% noise"
)


sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c2")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+2"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c1_c3")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 1+3"
)

sns.lineplot(
    test_fed_df[
        (test_fed_df.client == "c2_c3")
        & (test_fed_df.noise == noise)
    ], 
    x="samples", y="r2", linestyle=":", linewidth=2, errorbar=None, label="client 2+3"
)

    
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("noise_Y_fed_35p.pdf", bbox_inches="tight")
plt.show()

# %%
