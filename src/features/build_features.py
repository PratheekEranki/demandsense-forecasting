"""
Feature engineering pipeline for demand forecasting.
Generates lag features, rolling statistics, holiday flags, and calendar features.
"""

import pandas as pd
import numpy as np
import holidays
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load and parse the raw CSV, ensuring correct dtypes."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows, date range: {df.date.min()} → {df.date.max()}")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, quarter, week-of-year, and year features."""
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["dayofmonth"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Cyclical encoding for month and dayofweek (captures periodicity for tree models)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_holiday_features(df: pd.DataFrame, country: str = "US") -> pd.DataFrame:
    """Flag US federal holidays and days adjacent to them."""
    df = df.copy()
    years = df["date"].dt.year.unique().tolist()
    us_holidays = holidays.US(years=years)
    holiday_dates = set(us_holidays.keys())

    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in holiday_dates))
    df["is_day_before_holiday"] = df["date"].map(
        lambda d: int((d + pd.Timedelta(days=1)).date() in holiday_dates)
    )
    df["is_day_after_holiday"] = df["date"].map(
        lambda d: int((d - pd.Timedelta(days=1)).date() in holiday_dates)
    )
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    lags: List[int] = [7, 14, 21, 28, 91, 182, 364],
) -> pd.DataFrame:
    """
    Add lagged values of the target — grouped by store+item.
    Uses shift within group to prevent data leakage.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = (
            df.groupby(["store", "item"])[target_col]
            .shift(lag)
        )
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    windows: List[int] = [7, 14, 28, 91],
    lag: int = 28,  # roll on lag to avoid leakage
) -> pd.DataFrame:
    """
    Rolling mean, std, min, max — computed on a lagged series to prevent leakage.
    """
    df = df.copy()
    lagged = df.groupby(["store", "item"])[target_col].shift(lag)
    for w in windows:
        rolled = lagged.groupby([df["store"], df["item"]]).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_mean_{w}"] = rolled

        rolled_std = lagged.groupby([df["store"], df["item"]]).transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )
        df[f"rolling_std_{w}"] = rolled_std.fillna(0)

    return df


def add_expanding_features(
    df: pd.DataFrame, target_col: str = "sales"
) -> pd.DataFrame:
    """Expanding (cumulative) mean and std per store-item, lagged 28 days."""
    df = df.copy()
    lagged = df.groupby(["store", "item"])[target_col].shift(28)
    df["expanding_mean"] = lagged.groupby([df["store"], df["item"]]).transform(
        lambda x: x.expanding().mean()
    )
    df["expanding_std"] = lagged.groupby([df["store"], df["item"]]).transform(
        lambda x: x.expanding().std()
    ).fillna(0)
    return df


def add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Numerical day index as a proxy for global trend."""
    df = df.copy()
    min_date = df["date"].min()
    df["trend"] = (df["date"] - min_date).dt.days
    return df


def build_features(raw_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Full feature pipeline:
    load → calendar → holidays → lags → rolling → expanding → trend → save.
    """
    logger.info("Starting feature engineering pipeline...")
    df = load_raw_data(raw_path)

    # Check for missing values in sales
    missing = df["sales"].isna().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing sales values — forward-filling per store-item")
        df["sales"] = df.groupby(["store", "item"])["sales"].ffill().bfill()

    df = add_calendar_features(df)
    logger.info("Calendar features added")

    df = add_holiday_features(df)
    logger.info("Holiday features added")

    df = add_lag_features(df)
    logger.info("Lag features added")

    df = add_rolling_features(df)
    logger.info("Rolling features added")

    df = add_expanding_features(df)
    logger.info("Expanding features added")

    df = add_trend_feature(df)
    logger.info("Trend feature added")

    # Drop rows where lags are NaN (first ~364 rows per series)
    initial_rows = len(df)
    df = df.dropna(subset=["lag_364"]).reset_index(drop=True)
    logger.info(f"Dropped {initial_rows - len(df):,} rows with NaN lags → {len(df):,} rows remaining")

    if output_path:
        df.to_parquet(output_path, index=False)
        logger.info(f"Feature matrix saved to {output_path}")

    return df


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = os.getenv("RAW_DATA_PATH", "data/raw/train.csv")
    out = os.getenv("PROCESSED_DATA_PATH", "data/processed/features.parquet")
    os.makedirs("data/processed", exist_ok=True)
    df = build_features(raw, out)
    print(df.shape)
    print(df.dtypes)
