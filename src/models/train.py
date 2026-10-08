"""
Model training with MLflow tracking.
Trains: Seasonal Naïve baseline, XGBoost, LightGBM, and LSTM.
Uses time-series cross-validation (no leakage).
"""

import os
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dotenv import load_dotenv
from src.config import FEATURE_COLS, TARGET_COL

load_dotenv()
logger = logging.getLogger(__name__)


# ── Metrics ──────────────────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — robust to zero-sales days."""
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    w = wape(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "wape": round(w, 4)}


# ── Time-series CV splitter ───────────────────────────────────────────────────

def ts_cv_splits(df: pd.DataFrame, n_splits: int = 5):
    """
    Expanding-window time-series cross-validation.
    Yields (train_idx, val_idx) pairs with strictly forward-looking validation sets.
    """
    dates = df["date"].sort_values().unique()
    fold_size = len(dates) // (n_splits + 1)

    for i in range(1, n_splits + 1):
        cutoff_train = dates[i * fold_size - 1]
        cutoff_val = dates[min((i + 1) * fold_size - 1, len(dates) - 1)]
        train_mask = df["date"] <= cutoff_train
        val_mask = (df["date"] > cutoff_train) & (df["date"] <= cutoff_val)
        yield df.index[train_mask].tolist(), df.index[val_mask].tolist()


# ── Seasonal Naïve Baseline ───────────────────────────────────────────────────

def train_seasonal_naive(df: pd.DataFrame, experiment_name: str) -> dict:
    """Baseline: predict using the value from the same weekday 52 weeks ago (lag_364)."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="seasonal_naive"):
        all_metrics = []
        for train_idx, val_idx in ts_cv_splits(df):
            val = df.loc[val_idx]
            y_true = val[TARGET_COL].values
            y_pred = val["lag_364"].fillna(val["lag_182"]).fillna(0).values
            all_metrics.append(compute_metrics(y_true, y_pred))

        avg = {k: round(np.mean([m[k] for m in all_metrics]), 4) for k in ["mae", "rmse", "wape"]}
        mlflow.log_metrics(avg)
        mlflow.log_param("model_type", "seasonal_naive")
        logger.info(f"Seasonal Naïve CV: {avg}")
    return avg


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(df: pd.DataFrame, experiment_name: str) -> str:
    """Train XGBoost with time-series CV; log best model to MLflow registry."""
    import xgboost as xgb

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": int(os.getenv("RANDOM_STATE", 42)),
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda",
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_params(params)
        all_metrics = []

        for fold, (train_idx, val_idx) in enumerate(ts_cv_splits(df)):
            X_train = df.loc[train_idx, FEATURE_COLS]
            y_train = df.loc[train_idx, TARGET_COL]
            X_val = df.loc[val_idx, FEATURE_COLS]
            y_val = df.loc[val_idx, TARGET_COL]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = np.maximum(model.predict(X_val), 0)
            m = compute_metrics(y_val.values, y_pred)
            all_metrics.append(m)
            logger.info(f"XGB Fold {fold+1}: {m}")

        avg = {k: round(np.mean([m[k] for m in all_metrics]), 4) for k in ["mae", "rmse", "wape"]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg.items()})

        # Train final model on full dataset
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(df[FEATURE_COLS], df[TARGET_COL], verbose=False)

        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="demandsense-xgboost",
        )
        logger.info(f"XGBoost CV avg: {avg}")
        return run.info.run_id


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lightgbm(df: pd.DataFrame, experiment_name: str) -> str:
    """Train LightGBM with time-series CV; log to MLflow registry."""
    import lightgbm as lgb

    params = {
        "n_estimators": 500,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": int(os.getenv("RANDOM_STATE", 42)),
        "n_jobs": -1,
        "verbose": -1,
        "device": "gpu",
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="lightgbm") as run:
        mlflow.log_params(params)
        all_metrics = []

        for fold, (train_idx, val_idx) in enumerate(ts_cv_splits(df)):
            X_train = df.loc[train_idx, FEATURE_COLS]
            y_train = df.loc[train_idx, TARGET_COL]
            X_val = df.loc[val_idx, FEATURE_COLS]
            y_val = df.loc[val_idx, TARGET_COL]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
            )

            y_pred = np.maximum(model.predict(X_val), 0)
            m = compute_metrics(y_val.values, y_pred)
            all_metrics.append(m)
            logger.info(f"LGB Fold {fold+1}: {m}")

        avg = {k: round(np.mean([m[k] for m in all_metrics]), 4) for k in ["mae", "rmse", "wape"]}
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg.items()})

        # Final model — use last CV fold as eval set for early stopping
        last_train_idx, last_val_idx = list(ts_cv_splits(df))[-1]
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(
            df[FEATURE_COLS], df[TARGET_COL],
            eval_set=[(df.loc[last_val_idx, FEATURE_COLS], df.loc[last_val_idx, TARGET_COL])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="demandsense-lightgbm",
        )
        logger.info(f"LightGBM CV avg: {avg}")
        return run.info.run_id


# ── LSTM ──────────────────────────────────────────────────────────────────────

def build_lstm_sequences(df: pd.DataFrame, seq_len: int = 28):
    """Convert time-series data to (X, y) sequence pairs for LSTM."""
    import torch

    groups = []
    for (store, item), grp in df.groupby(["store", "item"]):
        grp = grp.sort_values("date")
        sales = grp[TARGET_COL].values.astype(np.float32)
        for i in range(seq_len, len(sales)):
            groups.append((sales[i - seq_len:i], sales[i]))

    X = np.stack([g[0] for g in groups])[:, :, None]  # (N, seq_len, 1)
    y = np.array([g[1] for g in groups])
    return torch.tensor(X), torch.tensor(y, dtype=torch.float32)


def train_lstm(df: pd.DataFrame, experiment_name: str, epochs: int = 20) -> str:
    """Train LSTM neural forecaster; log to MLflow."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    SEQ_LEN = 28
    HIDDEN = 64
    BATCH = 512
    LR = 1e-3

    # Use last year as validation (no leakage)
    cutoff = df["date"].max() - pd.Timedelta(days=91)
    train_df = df[df["date"] <= cutoff]
    val_df = df[df["date"] > cutoff]

    X_train, y_train = build_lstm_sequences(train_df, SEQ_LEN)
    X_val, y_val = build_lstm_sequences(val_df, SEQ_LEN)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, HIDDEN, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(1)

    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="lstm") as run:
        mlflow.log_params({"hidden": HIDDEN, "seq_len": SEQ_LEN, "epochs": epochs, "lr": LR})

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)

            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    val_preds.append(model(xb).cpu().numpy())
                    val_true.append(yb.numpy())

            y_pred = np.concatenate(val_preds)
            y_true = np.concatenate(val_true)
            m = compute_metrics(y_true, np.maximum(y_pred, 0))

            mlflow.log_metrics({"train_loss": train_loss / len(train_ds), **m}, step=epoch)
            logger.info(f"LSTM Epoch {epoch+1}/{epochs} — {m}")

        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name="demandsense-lstm")
        return run.info.run_id


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "demandsense-forecasting")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))
    os.makedirs("mlruns", exist_ok=True)

    features_path = os.getenv("PROCESSED_DATA_PATH", "data/processed/features.parquet")
    if not os.path.exists(features_path):
        logger.error(f"Features not found at {features_path}. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    logger.info(f"Loaded feature matrix: {df.shape}")

    train_seasonal_naive(df, experiment)
    xgb_run = train_xgboost(df, experiment)
    lgb_run = train_lightgbm(df, experiment)
    lstm_run = train_lstm(df, experiment, epochs=20)

    logger.info("All models trained and logged to MLflow.")
    logger.info(f"Run: mlflow ui --backend-store-uri {os.getenv('MLFLOW_TRACKING_URI')}")
