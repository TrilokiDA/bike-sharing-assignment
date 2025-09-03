import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def evaluate_baselines(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    # simple linear model baseline
    mae_lr = float("nan")
    try:
        lr = LinearRegression()
        lr.fit(X.fillna(0), y)
        pred_lr = lr.predict(X.fillna(0))
        mae_lr = mean_absolute_error(y, pred_lr)
    except Exception:
        pass
    return {"linear_regression_mae": float(mae_lr)}

def train_hgbr(X_train, y_train, X_valid, y_valid, random_state=42) -> Tuple[HistGradientBoostingRegressor, Dict]:
    model = HistGradientBoostingRegressor(
        max_depth=None,
        max_iter=400,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    return model, {"mae": float(mae)}

def holdout_split_chronological(X: pd.DataFrame, y: pd.Series, dt: pd.Series, holdout_frac=0.2):
    n = len(X)
    n_hold = int(n * holdout_frac)
    split = n - n_hold
    return (X.iloc[:split], y.iloc[:split], dt.iloc[:split],
            X.iloc[split:], y.iloc[split:], dt.iloc[split:])

def save_metrics(path: str, metrics: Dict):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
