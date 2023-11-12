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
from utils import RECORDS_RANDOM, SCALER, get_optimal_performance

# %%
# %config InlineBackend.figure_format='retina'

# %% [markdown]
# ## Get clients

# %%
n = SCALER.n_features_in_  # number of inputs, 8 for California housing
L = 200
W = np.random.randn(n, L)
bias = np.random.randn(1, L)

c1 = ClientNoiseHH(1, SCALER, RECORDS_RANDOM)
c2 = ClientNoiseHH(2, SCALER, RECORDS_RANDOM)
c3 = ClientNoiseHH(3, SCALER, RECORDS_RANDOM)

c1.init_elm(L, W, bias)
c2.init_elm(L, W, bias)
c3.init_elm(L, W, bias)

clients = [c1, c2, c3]

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

for noise in (-99, *np.logspace(-1, 1, num=5)):
    print(noise, end="  ")
    for idx, c in enumerate(clients):
        # logarithmic noise with no-noise special case
        c.noise_H = 0 if noise==-99 else noise  
        
        r2_c = get_optimal_performance(c, None, bsize[idx])
        for n, r2 in zip(bsize[idx], r2_c):
            test_data.append({"client": idx, "samples": n, "noise": noise, "r2": r2})

test_df = pd.DataFrame(test_data)

# %%
test_df

# %%
plt.rcParams['figure.figsize'] = [5, 3]

for idx in [0,1,2]:
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise > 0.1)], 
        x="samples", y="r2", units="noise", 
        color=".7", linewidth=1, estimator=None
    )

    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == 0.1)], 
        x="samples", y="r2", units="noise", 
        color="black", linestyle='dashed', linewidth=1.5, estimator=None, 
        label="noise=0.1" if idx==0 else None
    )

# no noise case
for idx, name in zip([0,1,2], ["client 1", "client 2", "client 3"]):
    sns.lineplot(
        test_df[(test_df.client == idx) & (test_df.noise == -99)], 
        x="samples", y="r2", linewidth=2.5, label=name
    )

plt.legend()
plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([-0.15, 0.75])
plt.xlabel("training samples")
plt.xscale("log")
plt.grid("major", axis="y")

plt.savefig("fig1.pdf", bbox_inches="tight")
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
