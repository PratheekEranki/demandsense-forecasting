# DemandSense: End-to-End Retail Demand Forecasting & Inventory Optimization Platform

An end-to-end ML system that forecasts daily retail demand at the store-item level using GPU-accelerated gradient boosted trees and LSTM neural networks, with MLflow experiment tracking, a FastAPI prediction service, Streamlit dashboard, and Evidently drift monitoring.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data & Features | Python 3.12, Pandas 2.2, NumPy 1.26, holidays |
| ML Models | XGBoost 2.0 (CUDA), LightGBM 4.3 (GPU), PyTorch 2.3 (CUDA LSTM) |
| Experiment Tracking | MLflow 2.13 (SQLite backend + model registry) |
| Serving | FastAPI 0.111, Uvicorn |
| Monitoring | Evidently 0.4 (data drift + prediction drift) |
| Dashboard | Streamlit 1.35, Plotly 5.22 |
| Testing | pytest 8.2, httpx (async API tests) |
| Infrastructure | Docker, Docker Compose, GitHub Actions CI |

---

## Architecture

```
Raw CSV (913K rows · 10 stores × 50 items × 5 years)
      │
      ▼
build_features.py       ← 35+ features: lags, rolling stats, holidays, cyclical calendar
      │
      ▼
data/processed/features.parquet  (731K rows × 36 columns)
      │
      ▼
train.py (GPU)  ──────► MLflow (experiments + model registry)
  ├── Seasonal Naïve        baseline
  ├── XGBoost (CUDA)        WAPE: 11.5%
  ├── LightGBM (GPU)        WAPE: 11.2%  ← best
  └── LSTM (CUDA)           WAPE: 12.6%
      │
      ▼
┌──────────────────────────┐   ┌───────────────────────┐
│  FastAPI  /predict       │   │  Streamlit Dashboard  │
│  /health  /metrics       │   │  forecast chart + PI  │
│  /stores  /items         │   │  inventory KPIs       │
└──────────────────────────┘   └───────────────────────┘
      │
      ▼
drift_report.py  ──► Evidently HTML reports + MLflow live metrics
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/PratheekEranki/demandsense-forecasting.git
cd demandsense-forecasting
python -m venv .venv
.venv/Scripts/activate    # Windows
pip install -r requirements.txt
cp .env.example .env
```

### 2. Download the data

Accept the competition rules at https://www.kaggle.com/competitions/demand-forecasting-kernels-only/rules, then:

```python
import kagglehub
path = kagglehub.competition_download('demand-forecasting-kernels-only')
# Copy train.csv and test.csv to data/raw/
```

### 3. Build features

Generates 35+ time-series features (lags, rolling statistics, holiday flags, cyclical calendar encoding) from raw sales data.

```bash
python -m src.features.build_features
# → data/processed/features.parquet (731K rows × 36 columns)
```

### 4. Train all models

Trains 4 models with GPU acceleration and 5-fold expanding-window time-series CV (no data leakage). All experiments are logged to MLflow.

```bash
python -m src.models.train
```

### 5. Launch MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# → http://localhost:5000
```

### 6. Start the API

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### 7. Launch the dashboard

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

### 8. Run monitoring

Generates Evidently drift reports and logs live performance metrics to MLflow.

```bash
python -m monitoring.drift_report
# → reports/figures/feature_distribution_drift_report.html
# → reports/figures/model_prediction_drift_report.html
```

### 9. Run tests

```bash
pytest tests/ -v
# 14 tests: unit (features, metrics) + integration (API endpoints)
```

---

## Docker

```bash
docker compose up --build
```

Services: API on `:8000`, Dashboard on `:8501`, MLflow on `:5000`.

---

## Model Performance

All models trained with GPU acceleration (NVIDIA RTX 4060) using 5-fold expanding-window time-series cross-validation.

| Model | MAE | RMSE | WAPE | Device | Role |
| --- | --- | --- | --- | --- | --- |
| LightGBM | 6.12 | 7.95 | 11.2% | GPU | Production model |
| XGBoost | 6.24 | 8.12 | 11.5% | CUDA | Alternative |
| LSTM | 6.70 | 8.91 | 12.6% | CUDA | Neural baseline |
| Seasonal Naïve | — | — | — | CPU | Statistical baseline |

**Live monitoring metrics** (Evidently on most recent 30-day window):

- MAE: 5.24 | RMSE: 6.78 | WAPE: 11.7%

---

## Feature Engineering

35+ features engineered per store-item time series, grouped into:

| Category | Features | Count |
|---|---|---|
| Calendar | dayofweek, dayofmonth, month, quarter, weekofyear, year, is_weekend | 7 |
| Cyclical | month_sin/cos, dow_sin/cos | 4 |
| Holiday | is_holiday, is_day_before_holiday, is_day_after_holiday | 3 |
| Lag | lag_7, lag_14, lag_21, lag_28, lag_91, lag_182, lag_364 | 7 |
| Rolling | rolling_mean/std for windows 7, 14, 28, 91 (lagged 28 days) | 8 |
| Expanding | expanding_mean, expanding_std (lagged 28 days) | 2 |
| Trend | linear day index from dataset start | 1 |

All lag and rolling features are computed on shifted (lagged) series to prevent data leakage.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/metrics` | Request count, error rate, avg latency |
| `GET` | `/stores` | List available store IDs |
| `GET` | `/items` | List available item IDs |
| `POST` | `/predict` | Generate demand forecast with prediction intervals and inventory recommendations |

**POST /predict** request body:
```json
{
  "store": 1,
  "item": 1,
  "horizon": 90,
  "model_name": "demandsense-lightgbm",
  "lead_time_days": 7
}
```

Supported models: `demandsense-lightgbm`, `demandsense-xgboost`, `demandsense-lstm`

**Response** includes:

- Daily predicted sales with 90% prediction intervals
- Inventory recommendations: safety stock, reorder point, total forecast (95% service level)

---

## Monitoring

Evidently-based drift detection generates two reports:

- **`feature_distribution_drift_report.html`** — Compares feature distributions between a 90-day reference window and the most recent 30 days using statistical tests (KS, chi-squared)
- **`model_prediction_drift_report.html`** — Scores both windows with the production model and checks for prediction distribution shift

Live performance metrics (MAE, RMSE, WAPE) are logged to MLflow for trend tracking.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | MLflow backend store |
| `MLFLOW_EXPERIMENT_NAME` | `demandsense-forecasting` | Experiment label |
| `RAW_DATA_PATH` | `data/raw/train.csv` | Input CSV path |
| `PROCESSED_DATA_PATH` | `data/processed/features.parquet` | Feature matrix path |
| `FORECAST_HORIZON` | `90` | Default forecast horizon (days) |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |
| `DASHBOARD_PORT` | `8501` | Streamlit port |

---

## Folder Structure

```
demandsense-forecasting/
├── api/                      # FastAPI prediction service
│   └── main.py               #   /predict, /health, /metrics, /stores, /items
├── dashboard/                # Streamlit interactive dashboard
│   └── app.py                #   forecast charts, inventory KPIs
├── data/
│   ├── raw/                  # train.csv, test.csv (gitignored)
│   └── processed/            # features.parquet (gitignored)
├── monitoring/               # Evidently drift detection
│   └── drift_report.py       #   feature + prediction drift reports
├── notebooks/                # Exploratory data analysis
│   └── 01_eda.ipynb          #   distributions, seasonality, lag analysis
├── reports/
│   └── figures/              # Generated charts and drift reports (gitignored)
├── src/
│   ├── features/
│   │   └── build_features.py #   35+ feature engineering pipeline
│   └── models/
│       ├── train.py          #   GPU training: XGBoost, LightGBM, LSTM
│       └── forecast.py       #   Inference + inventory optimization
├── tests/
│   ├── unit/                 # Feature and metric tests
│   └── integration/          # API endpoint tests
├── mlruns/                   # MLflow artifacts (gitignored)
├── .github/workflows/
│   └── ci.yml                # GitHub Actions: lint, test, build
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Key Accomplishments

- Engineered 35+ time-series features (lags, rolling statistics, holiday flags, cyclical calendar encoding) from 5 years of retail sales data across 500 store-item combinations
- Trained XGBoost and LightGBM with GPU acceleration and expanding-window time-series cross-validation (no leakage), achieving 11.2% WAPE on held-out data
- Implemented autoregressive LSTM neural forecaster with PyTorch CUDA for comparative evaluation; logged all experiments and registered models with MLflow
- Deployed real-time prediction API with FastAPI serving configurable-horizon demand forecasts with 90% prediction intervals and inventory reorder-point recommendations
- Built interactive Streamlit dashboard with Plotly for store/item-level forecast visualization, distribution analysis, and inventory KPIs
- Automated drift detection with Evidently, monitoring feature and prediction distributions against a rolling baseline with live metrics logged to MLflow
- Full test suite (14 tests) covering feature engineering, metric computation, and API integration

---

## Limitations & Future Work

- Dataset is simulated (Kaggle competition data) — real-world data would include promotional effects, pricing, and external signals
- LSTM currently trained on univariate series; a multivariate version with store metadata and feature inputs would improve accuracy
- Inventory model assumes constant lead time — a stochastic lead-time extension would improve safety stock estimates
- No online learning — model retraining is currently manual
- Prediction intervals use a simple Gaussian approximation; conformal prediction would give better coverage guarantees

---

## Data Source

Kaggle Store Item Demand Forecasting Challenge
https://www.kaggle.com/competitions/demand-forecasting-kernels-only
