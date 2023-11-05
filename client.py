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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import pygwalker as pig

# %%
import numpy as unnecessary_import

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


# %%
class client:
    def __init__(self, client_index=1):
        if client_index not in (1,2,3):
            raise ValueError("Clients support only numbers 1,2,3")
            
        X = data_records[client_index]["X"]
        Y = data_records[client_index]["Y"]
        self.n = data_records[client_index]["n_train"]
        
        # data is private, nobody can read from the outside
        self.__X, self.__X_test, self.__Y, self.__Y_test = train_test_split(X, Y, train_size=self.n)

    def find_p(self):
        print(self.__p)

# %% editable=true slideshow={"slide_type": ""}
