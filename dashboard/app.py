"""
Streamlit dashboard for DemandSense forecasting.
Features: store/item selector, forecast chart with PI, inventory KPIs,
model comparison table, and data quality summary.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DemandSense — Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 4px 0;
}
.kpi-label { color: #8b9ab0; font-size: 13px; font-weight: 500; }
.kpi-value { color: #e2e8f0; font-size: 28px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/inventory.png", width=60)
st.sidebar.title("DemandSense")
st.sidebar.caption("Retail Demand Forecasting · v1.0")

st.sidebar.divider()
store_id = st.sidebar.selectbox("Store", list(range(1, 11)), index=0)
item_id  = st.sidebar.selectbox("Item",  list(range(1, 51)), index=0)
horizon  = st.sidebar.slider("Forecast Horizon (days)", 7, 365, 90, step=7)
model_name = st.sidebar.selectbox(
    "Model",
    ["demandsense-lightgbm", "demandsense-xgboost", "demandsense-lstm"],
)
lead_time = st.sidebar.slider("Supplier Lead Time (days)", 1, 30, 7)
run_btn = st.sidebar.button("🔮 Generate Forecast", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("**About**")
st.sidebar.caption(
    "DemandSense uses gradient-boosted trees and LSTM neural networks "
    "to forecast daily retail demand, then computes inventory reorder points "
    "and safety stock at a 95% service level."
)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📦 DemandSense — Demand Forecasting & Inventory Optimization")

if not run_btn:
    st.info("Select a Store, Item, and horizon in the sidebar, then click **Generate Forecast**.")
    st.stop()

# ── API call ──────────────────────────────────────────────────────────────────
with st.spinner("Generating forecast..."):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={
                "store": store_id,
                "item": item_id,
                "horizon": horizon,
                "model_name": model_name,
                "lead_time_days": lead_time,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError:
        st.error("Cannot connect to the API. Start it with: `uvicorn api.main:app --reload`")
        st.stop()
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()

forecast_df = pd.DataFrame(data["forecast"])
forecast_df["date"] = pd.to_datetime(forecast_df["date"])
inv = data["inventory"]

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📅 Horizon",           f"{horizon} days")
col2.metric("📈 Avg Daily Demand",  f"{inv['avg_daily_demand']:.1f} units")
col3.metric("🛡️ Safety Stock",      f"{inv['safety_stock']:.0f} units")
col4.metric("🔔 Reorder Point",     f"{inv['reorder_point']:.0f} units")
col5.metric("📦 Total 90-day",      f"{inv['total_forecast_90d']:.0f} units")

st.divider()

# ── Forecast chart ────────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["upper_bound"],
    fill=None, mode="lines",
    line=dict(color="rgba(99,179,237,0)", width=0),
    showlegend=False, name="Upper CI",
))
fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["lower_bound"],
    fill="tonexty",
    fillcolor="rgba(99,179,237,0.15)",
    mode="lines",
    line=dict(color="rgba(99,179,237,0)", width=0),
    name="90% Prediction Interval",
))
fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["predicted_sales"],
    mode="lines",
    line=dict(color="#63B3ED", width=2.5),
    name="Predicted Sales",
))
# Reorder point line
fig.add_hline(
    y=inv["reorder_point"],
    line_dash="dash",
    line_color="#FC8181",
    annotation_text=f"Reorder Point ({inv['reorder_point']:.0f})",
    annotation_position="bottom right",
)
fig.update_layout(
    title=f"Store {store_id} · Item {item_id} — {horizon}-Day Demand Forecast ({model_name})",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    hovermode="x unified",
    template="plotly_dark",
    legend=dict(orientation="h", y=1.05),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ── Bottom row: distribution + inventory breakdown ────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Forecast Distribution")
    hist_fig = px.histogram(
        forecast_df,
        x="predicted_sales",
        nbins=30,
        color_discrete_sequence=["#63B3ED"],
        template="plotly_dark",
        labels={"predicted_sales": "Units"},
        title="Distribution of Daily Demand Predictions",
    )
    hist_fig.update_layout(height=300)
    st.plotly_chart(hist_fig, use_container_width=True)

with right:
    st.subheader("Inventory Recommendation")
    inv_table = pd.DataFrame([
        {"Metric": "Average Daily Demand", "Value": f"{inv['avg_daily_demand']:.2f} units"},
        {"Metric": "Demand Std Dev",       "Value": f"{inv['demand_std']:.2f} units"},
        {"Metric": "Lead Time",            "Value": f"{inv['lead_time_days']} days"},
        {"Metric": "Safety Stock (95% SL)","Value": f"{inv['safety_stock']:.0f} units"},
        {"Metric": "Reorder Point",        "Value": f"{inv['reorder_point']:.0f} units"},
        {"Metric": "Total 90-Day Forecast","Value": f"{inv['total_forecast_90d']:.0f} units"},
    ])
    st.dataframe(inv_table, use_container_width=True, hide_index=True)

# ── Raw forecast table ────────────────────────────────────────────────────────
with st.expander("📋 View raw forecast data"):
    display_df = forecast_df.copy()
    display_df.columns = ["Date", "Predicted Sales", "Lower Bound", "Upper Bound"]
    st.dataframe(display_df, use_container_width=True)

st.caption(f"Generated at: {data['generated_at']} · Model: {model_name}")
