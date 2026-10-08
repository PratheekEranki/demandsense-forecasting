"""
Inference module: load best MLflow model and generate store/item-level forecasts.
Also computes reorder points and safety stock from forecast + uncertainty.
"""

import os
import math
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
from dotenv import load_dotenv
from src.config import FEATURE_COLS

load_dotenv()
logger = logging.getLogger(__name__)

_LSTM_MODELS = {"demandsense-lstm"}
_VALID_MODELS = {"demandsense-lightgbm", "demandsense-xgboost", "demandsense-lstm"}
_model_cache: dict = {}


def load_model(model_name: str = "demandsense-lightgbm"):
    """Load MLflow model from registry with caching."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name not in _VALID_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}. Choose from {sorted(_VALID_MODELS)}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    uri = f"models:/{model_name}/latest"
    logger.info(f"Loading model: {model_name}")

    try:
        if model_name in _LSTM_MODELS:
            model = mlflow.pytorch.load_model(uri, map_location="cpu")
        else:
            model = mlflow.pyfunc.load_model(uri)
    except Exception:
        logger.warning(f"Registry lookup failed for {model_name}, searching runs...")
        model = _load_from_latest_run(model_name)

    _model_cache[model_name] = model
    return model


def _load_from_latest_run(model_name: str):
    """Fallback: find the latest training run and load from its artifacts."""
    experiment = mlflow.get_experiment_by_name(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "demandsense-forecasting")
    )
    if experiment is None:
        raise FileNotFoundError(f"No MLflow experiment found for {model_name}")

    run_name = model_name.replace("demandsense-", "")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise FileNotFoundError(f"No runs found for {model_name}")

    run_id = runs.iloc[0].run_id
    artifact_uri = f"runs:/{run_id}/model"
    logger.info(f"Loading {model_name} from run {run_id}")

    if model_name in _LSTM_MODELS:
        return mlflow.pytorch.load_model(artifact_uri, map_location="cpu")
    return mlflow.pyfunc.load_model(artifact_uri)


def generate_future_dates(start_date: str, horizon: int = 90) -> pd.DatetimeIndex:
    """Generate future date range for forecast horizon."""
    return pd.date_range(start=start_date, periods=horizon, freq="D")


def _predict_tree_recursive(model, df: pd.DataFrame, store: int, item: int, horizon: int) -> np.ndarray:
    """Row-by-row tree model prediction with recursive lag updates."""
    import holidays as hol

    series = df[(df["store"] == store) & (df["item"] == item)].sort_values("date")
    last_date = series["date"].max()
    future_dates = generate_future_dates(
        (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), horizon
    )

    sales_history = series["sales"].values.tolist()
    us_holidays = hol.US(years=future_dates.year.unique().tolist())
    holiday_dates = set(us_holidays.keys())
    min_date = df["date"].min()

    preds = []
    for d in future_dates:
        n = len(sales_history)
        row = {
            "store": store,
            "item": item,
            "dayofweek": d.dayofweek,
            "dayofmonth": d.day,
            "month": d.month,
            "quarter": d.quarter,
            "weekofyear": d.isocalendar()[1],
            "year": d.year,
            "is_weekend": int(d.dayofweek >= 5),
            "month_sin": np.sin(2 * np.pi * d.month / 12),
            "month_cos": np.cos(2 * np.pi * d.month / 12),
            "dow_sin": np.sin(2 * np.pi * d.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * d.dayofweek / 7),
            "is_holiday": int(d.date() in holiday_dates),
            "is_day_before_holiday": int((d + pd.Timedelta(days=1)).date() in holiday_dates),
            "is_day_after_holiday": int((d - pd.Timedelta(days=1)).date() in holiday_dates),
            "trend": (d - min_date).days,
            "lag_7":   sales_history[n - 7]   if n >= 7   else 0,
            "lag_14":  sales_history[n - 14]  if n >= 14  else 0,
            "lag_21":  sales_history[n - 21]  if n >= 21  else 0,
            "lag_28":  sales_history[n - 28]  if n >= 28  else 0,
            "lag_91":  sales_history[n - 91]  if n >= 91  else 0,
            "lag_182": sales_history[n - 182] if n >= 182 else 0,
            "lag_364": sales_history[n - 364] if n >= 364 else 0,
        }

        for w in [7, 14, 28, 91]:
            window = sales_history[max(0, n - w):]
            row[f"rolling_mean_{w}"] = np.mean(window) if window else 0
            row[f"rolling_std_{w}"] = np.std(window) if len(window) > 1 else 0

        row["expanding_mean"] = np.mean(sales_history)
        row["expanding_std"] = np.std(sales_history) if len(sales_history) > 1 else 0

        row_df = pd.DataFrame([row])
        pred = max(model.predict(row_df[FEATURE_COLS])[0], 0)
        preds.append(pred)
        sales_history.append(pred)

    return np.array(preds)


def _predict_lstm(
    model,
    df: pd.DataFrame,
    store: int,
    item: int,
    horizon: int,
) -> np.ndarray:
    """Autoregressive LSTM prediction: feed last 28 days, predict one step, slide window."""
    import torch

    SEQ_LEN = 28
    series = df[(df["store"] == store) & (df["item"] == item)].sort_values("date")
    history = series["sales"].values.astype(np.float32).tolist()

    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            seq = np.array(history[-SEQ_LEN:], dtype=np.float32).reshape(1, SEQ_LEN, 1)
            x = torch.tensor(seq)
            pred = max(model(x).item(), 0)
            preds.append(pred)
            history.append(pred)
    return np.array(preds)


def predict(
    df: pd.DataFrame,
    store: int,
    item: int,
    horizon: int = 90,
    model_name: str = "demandsense-lightgbm",
) -> pd.DataFrame:
    """
    Generate horizon-day demand forecast for a store-item combination.
    Returns DataFrame with date, predicted_sales, lower_bound, upper_bound.
    """
    model = load_model(model_name)

    series = df[(df["store"] == store) & (df["item"] == item)].sort_values("date")
    sigma = series["sales"].tail(28).std()
    last_date = series["date"].max()
    future_dates = generate_future_dates(
        (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), horizon
    )

    if model_name in _LSTM_MODELS:
        preds = _predict_lstm(model, df, store, item, horizon)
    else:
        preds = _predict_tree_recursive(model, df, store, item, horizon)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "store": store,
        "item": item,
        "predicted_sales": preds,
        "lower_bound": np.maximum(preds - 1.64 * sigma, 0),
        "upper_bound": preds + 1.64 * sigma,
    })

    return forecast_df[["date", "store", "item", "predicted_sales", "lower_bound", "upper_bound"]]


# ── Inventory optimization ────────────────────────────────────────────────────

def compute_inventory_recommendations(
    forecast_df: pd.DataFrame,
    lead_time_days: int = 7,
    service_level_z: float = 1.65,  # ~95% service level
) -> dict:
    """
    Compute reorder point and safety stock from demand forecast.

    reorder_point = avg_daily_demand * lead_time + safety_stock
    safety_stock  = z * sigma_demand * sqrt(lead_time)
    """
    avg_daily = forecast_df["predicted_sales"].mean()
    sigma = forecast_df["predicted_sales"].std()

    safety_stock = service_level_z * sigma * np.sqrt(lead_time_days)
    reorder_point = avg_daily * lead_time_days + safety_stock
    total_90d = forecast_df["predicted_sales"].sum()

    service_level_pct = round(0.5 * (1 + math.erf(service_level_z / math.sqrt(2))) * 100, 1)

    return {
        "avg_daily_demand": round(avg_daily, 2),
        "demand_std": round(sigma, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "total_forecast_90d": round(total_90d, 2),
        "lead_time_days": lead_time_days,
        "service_level_pct": service_level_pct,
    }
