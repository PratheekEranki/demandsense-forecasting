"""Unit tests for the feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    add_calendar_features,
    add_holiday_features,
    add_lag_features,
    add_rolling_features,
    add_trend_feature,
)


@pytest.fixture
def sample_df():
    """Minimal store-item time series for testing."""
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "store": [1] * 400 + [2] * 400,
        "item": [1] * 400 + [1] * 400,
        "sales": np.random.randint(5, 50, 800).astype(float),
    })
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def test_calendar_features_shape(sample_df):
    out = add_calendar_features(sample_df)
    assert "dayofweek" in out.columns
    assert "month_sin" in out.columns
    assert "month_cos" in out.columns
    assert "is_weekend" in out.columns
    assert out.shape[0] == sample_df.shape[0]


def test_is_weekend_correct(sample_df):
    out = add_calendar_features(sample_df)
    # dayofweek 5/6 should be weekend
    mask = out["dayofweek"] >= 5
    assert out.loc[mask, "is_weekend"].all() == True
    assert out.loc[~mask, "is_weekend"].sum() == 0


def test_holiday_features(sample_df):
    out = add_holiday_features(add_calendar_features(sample_df))
    assert "is_holiday" in out.columns
    assert out["is_holiday"].isin([0, 1]).all()
    # July 4th 2020 should be a holiday
    jul4 = out[out["date"] == "2020-07-04"]
    assert jul4["is_holiday"].values[0] == 1


def test_lag_features_no_leakage(sample_df):
    out = add_lag_features(sample_df, lags=[7])
    # lag_7 on row 0 within a group must be NaN (not enough history)
    first_rows = out.groupby(["store", "item"]).head(7)
    assert first_rows["lag_7"].isna().all()


def test_lag_values_correct(sample_df):
    out = add_lag_features(sample_df, lags=[7])
    # For store=1, item=1: row at index 7 should have lag_7 = row at index 0's sales
    grp = out[(out["store"] == 1) & (out["item"] == 1)].reset_index(drop=True)
    assert grp.loc[7, "lag_7"] == grp.loc[0, "sales"]


def test_rolling_features_non_negative(sample_df):
    df = add_lag_features(sample_df, lags=[28])
    out = add_rolling_features(df, windows=[7])
    assert (out["rolling_mean_7"].dropna() >= 0).all()


def test_trend_is_monotone(sample_df):
    out = add_trend_feature(sample_df)
    grp = out[(out["store"] == 1) & (out["item"] == 1)].sort_values("date")
    assert (grp["trend"].diff().dropna() >= 0).all()
