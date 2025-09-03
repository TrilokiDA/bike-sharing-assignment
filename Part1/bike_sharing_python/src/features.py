import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_weekend"] = out["weekday"].isin([0,6]).astype(int)  # 0=Sunday in dataset
    out["month"] = out["mnth"].astype(int)
    out["year"] = out["yr"].astype(int)
    out["hour"] = out["hr"].astype(int)
    # cyclic encodings for hour
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24.0)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24.0)
    return out

def add_lag_features(df: pd.DataFrame, lags=(1,2,24)) -> pd.DataFrame:
    out = df.copy()
    for l in lags:
        out[f"cnt_lag_{l}"] = out["cnt"].shift(l)
    return out

FEATURE_COLUMNS = [
    # calendar
    "workingday", "holiday", "is_weekend", "weekday", "month", "year",
    # time
    "hour", "hour_sin", "hour_cos",
    # weather
    "temp", "atemp", "hum", "windspeed",
    # weather state categories
    "weathersit", "season",
    # lags
    "cnt_lag_1", "cnt_lag_2", "cnt_lag_24",
]

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df1 = add_time_features(df)
    df2 = add_lag_features(df1)
    X = df2[FEATURE_COLUMNS].copy()
    y = df2["cnt"].copy()
    # drop rows where lag features are NaN (head of series)
    valid = ~X[["cnt_lag_1","cnt_lag_2","cnt_lag_24"]].isna().any(axis=1)
    return X[valid], y[valid], df2.loc[valid, "datetime"]
