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
from matplotlib import pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from client import ClientNoiseBoth

# %%
# %config InlineBackend.figure_format='retina'

# %% [markdown]
# # Prepare data

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

# %% [markdown]
# # Create clients

# %%
# records = data_records
records = data_records_random

# %%
scaler = RobustScaler().fit(X)

# %% [markdown]
# ## Get clients

# %%
# create ELM

n = X.shape[1]  # 8 inputs
L = 99
W = np.random.randn(n, L)
bias = np.random.randn(1, L)

c1 = ClientNoiseBoth(1, scaler, records)
c2 = ClientNoiseBoth(2, scaler, records)
c3 = ClientNoiseBoth(3, scaler, records)

for c in [c1, c2, c3]:
    c.init_elm(L, W, bias)
    c.noise_H = 0
    c.noise_Y = 0


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
r2_c3 = get_optimal_performance(c3, 300)


# %% [markdown]
# ## Extend single client with CLEAN data

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
clients = []
for _ in range(30):
    c1, c2, c3 = (ClientNoiseBoth(i, scaler, records) for i in (1,2,3))
    for c in (c1, c2, c3):
        c.init_elm(L, W, bias)
        c.noise_H = 0
        c.noise_Y = 0
    clients.append((c1, c2, c3))


# %%
def get_curves(c1, c2, c3, noise_h=0.1, noise_y=0.1):    
    # reset noise
    for c in (c1, c2, c3):
        c.noise_H = 0
        c.noise_Y = 0

    r2_c1 = get_optimal_performance(c1, 5)
    r2_c2 = get_optimal_performance(c2, 50)
    r2_c3 = get_optimal_performance(c3, 300)

    # add noise
    c3.noise_H = noise_h
    c3.noise_Y = noise_y
    r2_c1_c3 = get_optimal_performance_extended(c1, c3, 200)
    r2_c2_c3 = get_optimal_performance_extended(c2, c3, 300)
    
    c2.noise_H = noise_h
    c2.noise_Y = noise_y
    r2_c1_c2 = get_optimal_performance_extended(c1, c2, 50)

    return r2_c1, r2_c2, r2_c3, r2_c1_c2, r2_c1_c3, r2_c2_c3


# %% jupyter={"outputs_hidden": true}
results = [get_curves(c1, c2, c3, 0, 0) for c1,c2,c3 in clients]
r2c_c1, r2c_c2, r2c_c3, r2c_c1_c2, r2c_c1_c3, r2c_c2_c3 =  (np.mean([np.array(r[i]) for r in results], axis=0) for i in range(6))

# %%
plt.plot(np.arange(1, len(r2c_c1)+1)*5, r2c_c1)
plt.plot(np.arange(0, len(r2c_c1_c2))*50 + 100, r2c_c1_c2, '--', c="orange")
plt.plot(np.arange(0, len(r2c_c1_c3))*200 + 100, r2c_c1_c3, '--', c="red")

plt.plot(np.arange(1, len(r2c_c2)+1)*50, r2c_c2, c="orange")
plt.plot(np.arange(0, len(r2c_c2_c3))*300 + 1000, r2c_c2_c3, '--', c="red")

plt.plot(np.arange(1, len(r2c_c3)+1)*300, r2c_c3, c="red")

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([0, 0.75])
plt.xscale("log")
plt.show()

# %% [markdown]
# ## Extend single client with NOISY data

# %% jupyter={"outputs_hidden": true}
results = [get_curves(c1, c2, c3, 0.1, 0.1) for c1,c2,c3 in clients]
r2_c1, r2_c2, r2_c3, r2_c1_c2, r2_c1_c3, r2_c2_c3 =  (np.mean([np.array(r[i]) for r in results], axis=0) for i in range(6))

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(0, len(r2_c1_c2))*50 + 100, r2_c1_c2, '--', c="orange")
plt.plot(np.arange(0, len(r2_c1_c3))*200 + 100, r2_c1_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2, c="orange")
plt.plot(np.arange(0, len(r2_c2_c3))*300 + 1000, r2_c2_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3, c="red")

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([0, 0.75])
plt.xscale("log")
plt.show()

# %%

# %% jupyter={"outputs_hidden": true}
results2 = [get_curves(c1, c2, c3, 0.15, 0.15) for c1,c2,c3 in clients]
r2_c1, r2_c2, r2_c3, r2_c1_c2, r2_c1_c3, r2_c2_c3 =  (np.mean([np.array(r[i]) for r in results2], axis=0) for i in range(6))

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(0, len(r2_c1_c2))*50 + 100, r2_c1_c2, '--', c="orange")
plt.plot(np.arange(0, len(r2_c1_c3))*200 + 100, r2_c1_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2, c="orange")
plt.plot(np.arange(0, len(r2_c2_c3))*300 + 1000, r2_c2_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3, c="red")

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([0, 0.75])
plt.xscale("log")
plt.show()

# %% jupyter={"outputs_hidden": true}
results3 = [get_curves(c1, c2, c3, 0.2, 0.2) for c1,c2,c3 in clients]
r2_c1, r2_c2, r2_c3, r2_c1_c2, r2_c1_c3, r2_c2_c3 =  (np.mean([np.array(r[i]) for r in results3], axis=0) for i in range(6))

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(0, len(r2_c1_c2))*50 + 100, r2_c1_c2, '--', c="orange")
plt.plot(np.arange(0, len(r2_c1_c3))*200 + 100, r2_c1_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2, c="orange")
plt.plot(np.arange(0, len(r2_c2_c3))*300 + 1000, r2_c2_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3, c="red")

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([0, 0.75])
plt.xscale("log")
plt.show()

# %% [markdown]
# ## Compare with very noisy C3

# %%
res_noisy  = []

for _ in range(10):
    c3n = ClientNoiseBoth(3, scaler, records)
    c3n.init_elm(L, W, bias)
    c3n.noise_H = 0.2
    c3n.noise_Y = 0.2
    r2_c3_noisy = get_optimal_performance(c3n, 300)
    res_noisy.append(r2_c3_noisy)

r2_c3_noisy = np.mean([np.array(r1) for r1 in res_noisy], axis=0)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(0, len(r2_c1_c2))*50 + 100, r2_c1_c2, '--', c="orange")
plt.plot(np.arange(0, len(r2_c1_c3))*200 + 100, r2_c1_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2, c="orange")
plt.plot(np.arange(0, len(r2_c2_c3))*300 + 1000, r2_c2_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3, c="red")
plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3_noisy, ":", c="black")

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([0, 0.75])
plt.xscale("log")
plt.show()

# %% [markdown]
# ## Only Y hiding

# %% jupyter={"outputs_hidden": true}
results4 = [get_curves(c1, c2, c3, 0.0, 0.25) for c1,c2,c3 in clients]
r2_c1, r2_c2, r2_c3, r2_c1_c2, r2_c1_c3, r2_c2_c3 =  (np.mean([np.array(r[i]) for r in results4], axis=0) for i in range(6))

# %%
res_noisy_2  = []

for _ in range(10):
    c3n = ClientNoiseBoth(3, scaler, records)
    c3n.init_elm(L, W, bias)
    c3n.noise_H = 0.0
    c3n.noise_Y = 0.25
    r2_c3_noisy = get_optimal_performance(c3n, 300)
    res_noisy_2.append(r2_c3_noisy)

r2_c3_noisy_2 = np.mean([np.array(r1) for r1 in res_noisy_2], axis=0)

# %%
plt.plot(np.arange(1, len(r2_c1)+1)*5, r2_c1)
plt.plot(np.arange(0, len(r2_c1_c2))*50 + 100, r2_c1_c2, '--', c="orange")
plt.plot(np.arange(0, len(r2_c1_c3))*200 + 100, r2_c1_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c2)+1)*50, r2_c2, c="orange")
plt.plot(np.arange(0, len(r2_c2_c3))*300 + 1000, r2_c2_c3, '--', c="red")

plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3, c="red")
plt.plot(np.arange(1, len(r2_c3)+1)*300, r2_c3_noisy_2, ":", c="black")

plt.plot([0, 10000], [0, 0], '-k')
plt.ylim([0, 0.75])
plt.xscale("log")
plt.show()

# %%
