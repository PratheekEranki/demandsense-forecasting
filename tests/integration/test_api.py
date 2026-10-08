"""Integration tests for the FastAPI endpoints (uses TestClient — no running server needed)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


@pytest.fixture
def client():
    """Create a TestClient with mocked model and data."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    # Mock heavy dependencies so tests run without trained models
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([10.5] * 90)

    mock_df = pd.DataFrame({
        "store": [1] * 100,
        "item": [1] * 100,
        "date": pd.date_range("2017-01-01", periods=100),
        "sales": np.random.randint(5, 20, 100).astype(float),
        **{f"lag_{d}": np.random.rand(100) for d in [7, 14, 21, 28, 91, 182, 364]},
        **{f"rolling_mean_{w}": np.random.rand(100) for w in [7, 14, 28, 91]},
        **{f"rolling_std_{w}": np.random.rand(100) for w in [7, 14, 28, 91]},
        "expanding_mean": np.random.rand(100),
        "expanding_std": np.random.rand(100),
        "dayofweek": np.random.randint(0, 7, 100),
        "dayofmonth": np.random.randint(1, 28, 100),
        "month": np.random.randint(1, 12, 100),
        "quarter": np.random.randint(1, 4, 100),
        "weekofyear": np.random.randint(1, 52, 100),
        "year": [2017] * 100,
        "is_weekend": np.random.randint(0, 2, 100),
        "month_sin": np.random.rand(100),
        "month_cos": np.random.rand(100),
        "dow_sin": np.random.rand(100),
        "dow_cos": np.random.rand(100),
        "is_holiday": np.zeros(100),
        "is_day_before_holiday": np.zeros(100),
        "is_day_after_holiday": np.zeros(100),
        "trend": np.arange(100),
    })

    with patch("api.main.get_model", return_value=mock_model), \
         patch("api.main.get_data", return_value=mock_df), \
         patch("src.models.forecast.load_model", return_value=mock_model):
        from api.main import app
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "request_count" in resp.json()


def test_stores_endpoint(client):
    resp = client.get("/stores")
    assert resp.status_code == 200
    assert "stores" in resp.json()


def test_items_endpoint(client):
    resp = client.get("/items")
    assert resp.status_code == 200
    assert "items" in resp.json()
