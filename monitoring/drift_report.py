"""
Evidently-based monitoring: data drift, prediction drift, and model performance.
Generates HTML reports saved to reports/figures/.
"""

import os
import logging
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
from src.config import FEATURE_COLS

load_dotenv()
logger = logging.getLogger(__name__)


def load_reference_and_current(
    features_path: str,
    ref_days: int = 90,
    cur_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature data into reference (older) and current (recent) windows.
    Reference = data from [-(ref_days + cur_days)] to [-cur_days] ago.
    Current   = most recent cur_days of data.
    """
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()

    current_cutoff = max_date - pd.Timedelta(days=cur_days)
    ref_cutoff = max_date - pd.Timedelta(days=ref_days + cur_days)

    reference = df[(df["date"] >= ref_cutoff) & (df["date"] < current_cutoff)]
    current = df[df["date"] >= current_cutoff]

    logger.info(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")
    return reference, current


def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> str:
    """Generate Evidently data drift report comparing feature distributions."""
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[FEATURE_COLS],
        current_data=current[FEATURE_COLS],
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "feature_distribution_drift_report.html")
    report.save_html(out_path)
    logger.info(f"Data drift report saved to {out_path}")
    return out_path


def run_prediction_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_name: str = "demandsense-lightgbm",
    output_dir: str = "reports/figures",
) -> str:
    """
    Score both windows with the production model and check for prediction drift.
    Requires features (not labels) to be present in both DataFrames.
    """
    import mlflow.pyfunc
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

    ref_preds = pd.DataFrame({
        "prediction": np.maximum(model.predict(reference[FEATURE_COLS].fillna(0)), 0)
    })
    cur_preds = pd.DataFrame({
        "prediction": np.maximum(model.predict(current[FEATURE_COLS].fillna(0)), 0)
    })

    report = Report(metrics=[ColumnDriftMetric(column_name="prediction")])
    report.run(reference_data=ref_preds, current_data=cur_preds)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model_prediction_drift_report.html")
    report.save_html(out_path)
    logger.info(f"Prediction drift report saved to {out_path}")
    return out_path


def run_model_performance_report(
    current: pd.DataFrame,
    model_name: str = "demandsense-lightgbm",
    output_dir: str = "reports/figures",
) -> dict:
    """
    Score the current window and compute live MAE, RMSE, WAPE.
    Logs metrics to MLflow for trend tracking.
    """
    import mlflow.pyfunc

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

    X = current[FEATURE_COLS].fillna(0)
    y_true = current["sales"].values
    y_pred = np.maximum(model.predict(X), 0)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)

    metrics = {"live_mae": round(mae, 4), "live_rmse": round(rmse, 4), "live_wape": round(wape, 4)}

    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "demandsense-forecasting")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="monitoring_check"):
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model_name)

    logger.info(f"Live performance: {metrics}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    features_path = os.getenv("PROCESSED_DATA_PATH", "data/processed/features.parquet")

    if not os.path.exists(features_path):
        logger.error(f"Features not found at {features_path}. Run build_features.py first.")
        exit(1)

    ref, cur = load_reference_and_current(features_path)
    run_data_drift_report(ref, cur)
    run_prediction_drift_report(ref, cur)
    metrics = run_model_performance_report(cur)
    print("\nLive Performance Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
