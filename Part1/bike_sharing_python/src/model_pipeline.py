import os
import json
import argparse
import joblib
from datetime import datetime

import pandas as pd

from .data_io import load_hourly_df
from .features import build_feature_matrix
from .modeling import holdout_split_chronological, train_hgbr, save_metrics, evaluate_baselines

def run(data_dir: str, artifacts_dir: str, seed: int = 42):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "plots"), exist_ok=True)

    paths = data_dir +"/hour.csv"
    df = load_hourly_df(paths)

    # basic EDA plots
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        df.set_index("datetime")["cnt"].plot()
        plt.title("Hourly rentals over time")
        plt.savefig(os.path.join(artifacts_dir, "plots", "timeseries_cnt.png"))
        plt.close()

        plt.figure()
        df["hr"].value_counts().sort_index().plot(kind="bar")
        plt.title("Count by hour of day")
        plt.savefig(os.path.join(artifacts_dir, "plots", "hist_hour.png"))
        plt.close()

        plt.figure()
        df.groupby("hr")["cnt"].mean().plot()
        plt.title("Average cnt by hour")
        plt.savefig(os.path.join(artifacts_dir, "plots", "avg_cnt_by_hour.png"))
        plt.close()
    except Exception as e:
        print("Plotting failed (non-fatal):", e)

    X, y, dt = build_feature_matrix(df)

    X_tr, y_tr, dt_tr, X_va, y_va, dt_va = holdout_split_chronological(X, y, dt, holdout_frac=0.2)

    # Baseline(s)
    baselines = evaluate_baselines(X_va, y_va)

    # Train chosen model
    model, res = train_hgbr(X_tr, y_tr, X_va, y_va, random_state=seed)

    # Save model
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, os.path.join(artifacts_dir, "model.joblib"))

    # Save metrics
    metrics = {
        "metric": "MAE",
        "mae": res["mae"],
        "baselines": baselines,
        "n_train": int(len(X_tr)),
        "n_valid": int(len(X_va)),
        "seed": seed,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_metrics(os.path.join(artifacts_dir, "metrics.json"), metrics)
    print(json.dumps(metrics, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.data_dir, args.artifacts_dir, args.seed)

if __name__ == "__main__":
    main()
