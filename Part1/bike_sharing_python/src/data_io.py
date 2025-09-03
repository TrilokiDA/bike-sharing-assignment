import pandas as pd


def load_hourly_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # build a proper datetime index
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df
