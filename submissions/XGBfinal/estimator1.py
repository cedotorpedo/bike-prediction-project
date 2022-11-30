from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb


# The date encoder stays the same, we changed nothing here
def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])



### EXPLANATION OF EXTERNAL DATA
## We used 5 variables for external data. 
## 3 of them come from the Weather data set "etat_sol", "ww" and "rr1".
## The "conf" variable indicates (1 or 0) whether there was confinement/lockdown on that day. 
    # "Confinement" is to be understood loosely: the entire period from the 26/10/2020 until the 23/06/2021 is considered confinement.
## The "hourly" variable indicates whether there was a curfew in place on a given hour.
#Apart from that and the name of the csv file nothing about the function was changed.

def _merge_external_data(X):
    file_path = Path(__file__).parent / "final.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date","conf","hourly","ww","rr1","etat_sol"]].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name", "etat_sol", "conf", "hourly", "ww"]
    numerical_cols = ["rr1"]
    

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ("num", 'passthrough', numerical_cols) 
        ]
    )

    # We picked a XGB Regressor for our regressor
    regressor = xgb.XGBRegressor(max_depth=8, objective='reg:squarederror', learning_rate=0.2,n_estimators=110)


    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor
    )

    return pipe
