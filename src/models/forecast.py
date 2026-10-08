"""
Inference module: load best MLflow model and generate store/item-level forecasts.
Also computes reorder points and safety stock from forecast + uncertainty.
"""

import os
import logging
import numpy as np
import pandas as pd
import mlflow.pyfunc
import mlflow.pytorch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "store", "item", "dayofweek", "dayofmonth", "month", "quarter",
    "weekofyear", "year", "is_weekend", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "is_holiday", "is_day_before_holiday",
    "is_day_after_holiday", "trend",
    "lag_7", "lag_14", "lag_21", "lag_28", "lag_91", "lag_182", "lag_364",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_mean_91",
    "rolling_std_7", "rolling_std_14", "rolling_std_28", "rolling_std_91",
    "expanding_mean", "expanding_std",
]


# Map registered model names → relative artifact paths (avoids cross-OS registry URI issues)
_MODEL_PATHS = {
    "demandsense-lightgbm": "mlruns/1/8cbf0ac803c542f2a8a5be2690936a5f/artifacts/model",
    "demandsense-xgboost":  "mlruns/1/38b7c04e584c4243be5b968aa16c34ef/artifacts/model",
    "demandsense-lstm":     "mlruns/1/1bd18ed4d82246959a13af4e407a2c78/artifacts/model",
}

_LSTM_MODELS = {"demandsense-lstm"}


def load_model(model_name: str = "demandsense-lightgbm", stage: str = "None"):
    """Load MLflow model by direct artifact path (works cross-platform)."""
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parents[2]
    rel_path = _MODEL_PATHS.get(model_name)
    if rel_path is None:
        raise ValueError(f"Unknown model name: {model_name!r}. Choose from {list(_MODEL_PATHS)}")
    artifact_path = project_root / rel_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path}")
    logger.info(f"Loading model: {model_name} from {artifact_path}")
    if model_name in _LSTM_MODELS:
        return mlflow.pytorch.load_model(str(artifact_path), map_location="cpu")
    return mlflow.pyfunc.load_model(str(artifact_path))


def generate_future_dates(start_date: str, horizon: int = 90) -> pd.DatetimeIndex:
    """Generate future date range for forecast horizon."""
    return pd.date_range(start=start_date, periods=horizon, freq="D")


def build_forecast_frame(
    df: pd.DataFrame,
    store: int,
    item: int,
    horizon: int = 90,
) -> pd.DataFrame:
    """
    Build a feature-ready DataFrame for a future forecast window.
    Uses last known lags from training data, then propagates recursively.
    """
    import holidays as hol

    series = df[(df["store"] == store) & (df["item"] == item)].sort_values("date")
    last_date = series["date"].max()
    future_dates = generate_future_dates(
        (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), horizon
    )

    rows = []
    sales_history = series["sales"].values.tolist()

    us_holidays = hol.US(years=future_dates.year.unique().tolist())
    holiday_dates = set(us_holidays.keys())

    for i, d in enumerate(future_dates):
        n = len(sales_history)
        row = {
            "date": d,
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
            "trend": (d - df["date"].min()).days,
            # Lags — look back into history + previously predicted values
            "lag_7":   sales_history[n - 7]   if n >= 7   else np.nan,
            "lag_14":  sales_history[n - 14]  if n >= 14  else np.nan,
            "lag_21":  sales_history[n - 21]  if n >= 21  else np.nan,
            "lag_28":  sales_history[n - 28]  if n >= 28  else np.nan,
            "lag_91":  sales_history[n - 91]  if n >= 91  else np.nan,
            "lag_182": sales_history[n - 182] if n >= 182 else np.nan,
            "lag_364": sales_history[n - 364] if n >= 364 else np.nan,
        }

        # Rolling statistics on available history
        for w in [7, 14, 28, 91]:
            window = sales_history[max(0, n - w):]
            row[f"rolling_mean_{w}"] = np.mean(window) if window else 0
            row[f"rolling_std_{w}"] = np.std(window) if len(window) > 1 else 0

        row["expanding_mean"] = np.mean(sales_history)
        row["expanding_std"] = np.std(sales_history) if len(sales_history) > 1 else 0

        rows.append(row)
        # Placeholder for the next iteration's lag (will be filled by model prediction)
        sales_history.append(0)  # updated after prediction below

    return pd.DataFrame(rows)


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
            pred = model(x).item()
            pred = max(pred, 0)
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
        forecast_df = pd.DataFrame({
            "date": future_dates,
            "store": store,
            "item": item,
        })
    else:
        forecast_df = build_forecast_frame(df, store, item, horizon)
        X = forecast_df[FEATURE_COLS].fillna(0)
        preds = np.maximum(model.predict(X), 0)

    forecast_df["predicted_sales"] = preds
    forecast_df["lower_bound"] = np.maximum(preds - 1.64 * sigma, 0)
    forecast_df["upper_bound"] = preds + 1.64 * sigma

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

    return {
        "avg_daily_demand": round(avg_daily, 2),
        "demand_std": round(sigma, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "total_forecast_90d": round(total_90d, 2),
        "lead_time_days": lead_time_days,
        "service_level_pct": round(service_level_z * 100 / 1.65 * 95 / 100, 1),
    }
