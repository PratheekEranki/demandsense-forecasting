"""
FastAPI prediction service for DemandSense.
Endpoints: /predict, /health, /metrics, /inventory
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────

_model = None
_data: Optional[pd.DataFrame] = None
_request_count = 0
_error_count = 0
_latencies = []


def get_model():
    global _model
    if _model is None:
        from src.models.forecast import load_model
        _model = load_model()
    return _model


def get_data() -> pd.DataFrame:
    global _data
    if _data is None:
        path = os.getenv("PROCESSED_DATA_PATH", "data/processed/features.parquet")
        if os.path.exists(path):
            _data = pd.read_parquet(path)
            _data["date"] = pd.to_datetime(_data["date"])
        else:
            raise RuntimeError(f"Feature data not found at {path}. Run build_features.py first.")
    return _data


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and data on startup...")
    try:
        get_model()
        get_data()
        logger.info("Model and data loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load model/data: {e}")
    yield
    logger.info("Shutting down DemandSense API")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DemandSense Forecasting API",
    description="Retail demand forecasting and inventory optimization service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    store: int = Field(..., ge=1, le=10, description="Store ID (1–10)")
    item: int = Field(..., ge=1, le=50, description="Item ID (1–50)")
    horizon: int = Field(90, ge=1, le=365, description="Forecast horizon in days")
    model_name: str = Field("demandsense-lightgbm", description="Registered MLflow model name")
    lead_time_days: int = Field(7, ge=1, le=90, description="Supplier lead time for inventory calc")


class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float
    lower_bound: float
    upper_bound: float


class InventoryRec(BaseModel):
    avg_daily_demand: float
    demand_std: float
    safety_stock: float
    reorder_point: float
    total_forecast_90d: float
    lead_time_days: int
    service_level_pct: float


class ForecastResponse(BaseModel):
    store: int
    item: int
    horizon: int
    model_name: str
    forecast: list[ForecastPoint]
    inventory: InventoryRec
    generated_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "demandsense-api", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    avg_latency = round(np.mean(_latencies[-100:]), 3) if _latencies else 0
    return {
        "request_count": _request_count,
        "error_count": _error_count,
        "avg_latency_ms": avg_latency,
        "error_rate": round(_error_count / max(_request_count, 1), 4),
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    global _request_count, _error_count, _latencies
    _request_count += 1
    t0 = time.perf_counter()

    try:
        from src.models.forecast import predict as run_forecast, compute_inventory_recommendations
        df = get_data()

        # Validate store/item exist in data
        available_stores = df["store"].unique().tolist()
        available_items = df["item"].unique().tolist()
        if req.store not in available_stores:
            raise HTTPException(status_code=400, detail=f"Store {req.store} not in dataset")
        if req.item not in available_items:
            raise HTTPException(status_code=400, detail=f"Item {req.item} not in dataset")

        forecast_df = run_forecast(df, req.store, req.item, req.horizon, req.model_name)
        inv = compute_inventory_recommendations(forecast_df, req.lead_time_days)

        forecast_points = [
            ForecastPoint(
                date=row["date"].strftime("%Y-%m-%d"),
                predicted_sales=round(row["predicted_sales"], 2),
                lower_bound=round(row["lower_bound"], 2),
                upper_bound=round(row["upper_bound"], 2),
            )
            for _, row in forecast_df.iterrows()
        ]

        _latencies.append((time.perf_counter() - t0) * 1000)
        return ForecastResponse(
            store=req.store,
            item=req.item,
            horizon=req.horizon,
            model_name=req.model_name,
            forecast=forecast_points,
            inventory=InventoryRec(**inv),
            generated_at=pd.Timestamp.now().isoformat(),
        )

    except HTTPException:
        _error_count += 1
        raise
    except Exception as e:
        _error_count += 1
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stores")
def list_stores():
    df = get_data()
    return {"stores": sorted(df["store"].unique().tolist())}


@app.get("/items")
def list_items():
    df = get_data()
    return {"items": sorted(df["item"].unique().tolist())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
