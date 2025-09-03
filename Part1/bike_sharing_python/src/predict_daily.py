import os
import argparse
import joblib
import pandas as pd

from .data_io import load_hourly_df
from .features import build_feature_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD date to predict for (24 hours)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Load model
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Load data to build lags up to the target date
    data_dir = args.data_dir
    paths = data_dir + "/hour.csv"
    df = load_hourly_df(paths)

    target_date = pd.to_datetime(args.date).normalize()

    # We will predict for 24 hours of target_date; features require prior history for lags
    X, y, dt = build_feature_matrix(df)

    # select rows corresponding to the target_date 24 hours
    mask = (dt.dt.date == target_date.date())
    X_day = X.loc[mask, feature_columns]

    if X_day.empty:
        raise ValueError("No rows found for the specified date. Check the dataset range.")

    preds = model.predict(X_day)

    out_df = pd.DataFrame({
        "datetime": dt.loc[mask].values,
        "pred_cnt": preds
    })

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_df.to_csv(args.out, index=False)
    else:
        print(out_df.to_csv(index=False))

if __name__ == "__main__":
    main()
