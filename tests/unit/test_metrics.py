"""Unit tests for metric computation."""

import numpy as np
import pytest
from src.models.train import compute_metrics, wape


def test_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    m = compute_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["wape"] == pytest.approx(0.0, abs=1e-6)


def test_wape_known_value():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([80.0, 120.0])
    result = wape(y_true, y_pred)
    assert result == pytest.approx(0.2, abs=1e-4)


def test_metrics_non_negative():
    y_true = np.random.rand(100) * 50
    y_pred = np.random.rand(100) * 50
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] >= 0
    assert m["rmse"] >= 0
    assert m["wape"] >= 0
