# Bike Sharing: Hourly Utilization (cnt)

This repo contains:
- Plots, metrics, trained model, pred (`artifacts/`)
- Reproducible code under `src/`
- Data under `data/`
- A minimal daily prediction script to mimic a production job (`src/predict_daily.py`).

## Quickstart

```bash
# 1) Create env and install deps
pip install -r requirements.txt

# 2) Run end-to-end (downloads data -> EDA plots -> train -> evaluate -> save model)
python -m src.model_pipeline --data-dir data --artifacts-dir artifacts --seed 42

# 3) See evaluation
cat artifacts/metrics.json

# 4) Generate next-day hourly predictions (example date is 2012-12-20; adjust as needed)
python -m src.predict_daily --model-path artifacts/model.joblib --date 2012-12-20 --data-dir data --out artifacts/preds_2012-12-20.csv
```

## Next Steps
- Create API
    - Route for hourly/weekly/monthly forecasting