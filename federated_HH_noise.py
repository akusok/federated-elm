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

from client import ClientNoiseHH
from utils import (
    RECORDS_RANDOM, 
    SCALER, 
    get_optimal_performance, 
    get_optimal_performance_extended
)

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
    clients = []
    for idx in (1,2,3):
        client = ClientNoiseHH(idx, SCALER, RECORDS_RANDOM)
        client.init_elm(L, W, bias)
        clients.append(client)
    return clients


clients = get_clients()

# %%
# generate batch sizes unifromly for log scale

bsize1 = np.logspace(1, 2, num=15).astype(int)
bsize2 = np.logspace(1.5, 3, num=15).astype(int)
bsize3 = np.logspace(2, 4, num=15).astype(int)
bsize = [bsize1, bsize2, bsize3]

# %% [markdown]
# ## Find independent performance in batches

# %% [markdown]
# ## Get noise effect data

# %%
test_data = []

for run in range(10):
    print(run, end=":  ")
    clients = get_clients()
    
    for noise in (-99, 0.2, *np.logspace(-1, 1, num=5)):
        print(noise, end="  ")
        for idx, c in enumerate(clients):
            # logarithmic noise with no-noise special case
            c.noise_H = 0 if noise==-99 else noise  
            
            r2_c = get_optimal_performance(c, None, bsize[idx])
            for n, r2 in zip(bsize[idx], r2_c):
                test_data.append({"run": run, "client": idx, "samples": n, "noise": noise, "r2": r2})
    print()

test_df = pd.DataFrame(test_data)
test_df.to_csv("noise_HH.csv")

# %%
plt.rcParams['figure.figsize'] = [4, 3]

for idx in [0,1,2]:
    sns.lineplot(
        test_df[
            (test_df.client == idx) 
            & (test_df.noise > 0.2)
            & (test_df.run == 1)
        ], 
        x="samples", y="r2", units="noise", 
        color=".7", linewidth=1, estimator=None
    )

# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == -99)], 
        x="samples", y="r2", linewidth=2.5, label=name
    )
    
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == 0.2)], 
        x="samples", y="r2", errorbar=None,
        color="black", linestyle=':', linewidth=1.5, 
        label="noise=0.2" if idx==2 else None 
    )

plt.legend()
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xlabel("training samples")
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("noise_HH.pdf", bbox_inches="tight")
plt.show()

# %%
plt.rcParams['figure.figsize'] = [4, 3]

for idx in [0,1,2]:
    sns.lineplot(
        test_df[
            (test_df.client == idx) 
            & (test_df.noise > 0.1)
            & (test_df.run == 1)
        ], 
        x="samples", y="r2", units="noise", 
        color=".7", linewidth=1, estimator=None
    )

# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == -99)], 
        x="samples", y="r2", linewidth=2.5, label=name
    )
    
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == 0.31622776601683794)], 
        x="samples", y="r2", errorbar=None,
        color="black", linestyle=':', linewidth=1.5, 
        label="noise=0.33" if idx==2 else None 
    )

plt.legend()
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xlabel("training samples")
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("noise_HH_2.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Look at HH values with "acceptable" noise of 10**-1

# %%
np.set_printoptions(precision=3)

# %%
c1 = clients[0]

# print("no noise")
c1.noise_H = 0
print(c1.H[:5,:5])

print()
# print("noise of 0.2")
c1.noise_H = 0.2
print(c1.H[:5,:5])

# %%
