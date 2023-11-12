import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# California data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = data.target

# Split in geographical blocks
RECORDS_GEOGRAPHICAL = {
    1: {"n_train": 100, "X": X[X.Latitude > 40], "Y": Y[X.Latitude > 40]},
    2: {"n_train": 1000, "X": X[X.Longitude > -118], "Y": Y[X.Longitude > -118]},
    3: {"n_train": 10_000, "X": X[(X.Longitude <= -118) & (X.Latitude <= 40)], "Y": Y[(X.Longitude <= -118) & (X.Latitude <= 40)]},
}

# split randomly, same number of samples per client
X1, X23, Y1, Y23 = train_test_split(X, Y, train_size = RECORDS_GEOGRAPHICAL[1]["X"].shape[0])
X2, X3, Y2, Y3 = train_test_split(X23, Y23, train_size = RECORDS_GEOGRAPHICAL[2]["X"].shape[0])

RECORDS_RANDOM = {
    1: {"n_train": 100, "X": X1, "Y": Y1},
    2: {"n_train": 1000, "X": X2, "Y": Y2},
    3: {"n_train": 10_000, "X": X3, "Y": Y3},
}

SCALER = RobustScaler().fit(X)


# functions
def get_optimal_performance(client, batch_size=100, batches=None):
    L2_list = np.logspace(-5, 3, num=9)
    HH = 0
    HY = 0
    r2_test = []

    for hh, hy in client.batch_data(batch_size, batches):
        HH += hh
        HY += hy

        # find best performance
        r2 = -999999
        for l2 in L2_list:
            client.B = np.linalg.lstsq(HH + l2 * np.eye(HH.shape[0]), HY, rcond=None)[0]
            r2 = max(r2, client.r2)
        r2_test.append(r2)

    return r2_test

def get_optimal_performance_extended(client, fed_client, fed_batches):
    L2_list = np.logspace(-5, 3, num=9)
    HH = client.HH
    HY = client.HY
    r2_extended = []

    r2 = -999
    for l2 in L2_list:
        client.B = np.linalg.lstsq(HH + l2 * np.eye(HH.shape[0]), HY, rcond=None)[0]
        r2 = max(r2, client.r2)
    r2_extended.append(r2)

    # streaming data from collab client, evaluate for the original client
    for hh, hy in fed_client.batch_data(None, batches=fed_batches):
        HH += hh
        HY += hy

        # find best performance
        r2 = -999
        for l2 in L2_list:
            client.B = np.linalg.lstsq(HH + l2 * np.eye(HH.shape[0]), HY, rcond=None)[0]
            r2 = max(r2, client.r2)
        r2_extended.append(r2)

    return r2_extended
