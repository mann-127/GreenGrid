"""
GreenGrid — Streamlit Dashboard
=================================
Interactive visualisation of forecasts, dispatch decisions, and metrics.

Launch with::

    streamlit run greengrid/dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from greengrid.data.generator import generate_dataset
from greengrid.data.preprocessing import prepare_data
from greengrid.evaluation.metrics import compute_all_metrics, curtailment_reduction
from greengrid.models.baseline import fit_baseline, moving_average_forecast
from greengrid.optimizer.battery import BatteryConfig
from greengrid.optimizer.dispatch import ForecastDispatch, NaiveDispatch
from greengrid.settings import CFG

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenGrid — Smart Grid Controller",
    page_icon=":zap:",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_data():
    """Generate and cache data."""
    df = generate_dataset(CFG)
    data = prepare_data(df, CFG)
    return df, data


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/wind-turbine.png", width=80)
st.sidebar.title("GreenGrid")
st.sidebar.markdown("**AI-Driven Smart Grid Controller**")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Forecasting", "Battery & Dispatch", "Metrics"],
)

df, data = load_data()

# ══════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ══════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("GreenGrid — Renewable Energy Dashboard")
    st.markdown(
        """
    This dashboard visualises the AI-driven smart grid controller that
    predicts solar & wind energy production and optimises battery dispatch
    to **minimise curtailment** and **maximise revenue**.
    """
    )

    col1, col2, col3, col4 = st.columns(4)
    wind_cfg = CFG["data_generation"]["wind"]
    solar_cfg = CFG["data_generation"]["solar"]
    batt_cfg = CFG["data_generation"]["battery"]

    col1.metric(
        "Wind Farm",
        f"{wind_cfg['num_turbines'] * wind_cfg['rated_capacity_mw']:.0f} MW",
    )
    col2.metric("Solar Farm", f"{solar_cfg['capacity_mw']:.0f} MW")
    col3.metric("Battery", f"{batt_cfg['capacity_mwh']:.0f} MWh")
    col4.metric("Data Points", f"{len(df):,}")

    st.subheader("Raw Data Preview")
    st.dataframe(df.tail(48), use_container_width=True)

    # ── Time-series plot ─────────────────────────────────────────────
    st.subheader("Energy Production Over Time")
    sample = df.set_index("timestamp").iloc[-720:]  # last 30 days

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Wind Power (MW)", "Solar Power (MW)"],
    )
    fig.add_trace(
        go.Scatter(
            x=sample.index,
            y=sample["wind_power_mw"],
            name="Wind",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sample.index,
            y=sample["solar_power_mw"],
            name="Solar",
            line=dict(color="#ff7f0e"),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: Forecasting
# ══════════════════════════════════════════════════════════════════════
elif page == "Forecasting":
    st.title("24-Hour Probabilistic Forecast")

    # Baseline
    baseline = fit_baseline(data.train.y, data.train.y, data.val.y, data.val.y)
    test_pred = moving_average_forecast(
        data.test.y,
        data.test.y.shape[1],
        min(baseline.best_window, data.test.y.shape[1]),
    )

    # Pick a sample
    sample_idx = st.slider("Sample index", 0, len(data.test.y) - 1, 0)

    actual = data.target_scaler.inverse_transform(data.test.y[sample_idx].reshape(-1, 2))
    predicted = data.target_scaler.inverse_transform(test_pred[sample_idx].reshape(-1, 2))

    target_names = CFG["preprocessing"]["target_columns"]
    for t, name in enumerate(target_names):
        fig = go.Figure()
        hours = list(range(1, 25))
        fig.add_trace(go.Scatter(x=hours, y=actual[:, t], name="Actual", mode="lines+markers"))
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=predicted[:, t],
                name="Baseline MA",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
        # Confidence bands (from baseline quantiles)
        for q_lo, q_hi, color in [
            (0.05, 0.95, "rgba(31,119,180,0.15)"),
            (0.25, 0.75, "rgba(31,119,180,0.3)"),
        ]:
            if q_lo in baseline.quantiles and q_hi in baseline.quantiles:
                lo = data.target_scaler.inverse_transform(baseline.quantiles[q_lo][sample_idx].reshape(-1, 2))[:, t]
                hi = data.target_scaler.inverse_transform(baseline.quantiles[q_hi][sample_idx].reshape(-1, 2))[:, t]
                fig.add_trace(
                    go.Scatter(
                        x=hours + hours[::-1],
                        y=list(hi) + list(lo[::-1]),
                        fill="toself",
                        fillcolor=color,
                        line=dict(width=0),
                        name=f"PI {q_lo}-{q_hi}",
                    )
                )
        fig.update_layout(
            title=f"{name} — 24h Forecast",
            xaxis_title="Hour Ahead",
            yaxis_title="MW",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: Battery & Dispatch
# ══════════════════════════════════════════════════════════════════════
elif page == "Battery & Dispatch":
    st.title("Battery Dispatch Simulation")

    horizon = 24
    rng = np.random.default_rng(42)
    renewable = 80 + 40 * np.sin(2 * np.pi * np.arange(horizon) / 24) + rng.normal(0, 10, horizon)
    renewable = np.clip(renewable, 0, 200)
    demand = 400 + 150 * np.sin(2 * np.pi * np.arange(horizon) / 24) + rng.normal(0, 20, horizon)
    price = np.array(
        [
            (
                CFG["optimizer"]["electricity_peak_price_mwh"]
                if h in CFG["optimizer"]["peak_hours"]
                else CFG["optimizer"]["electricity_sell_price_mwh"]
            )
            for h in range(horizon)
        ]
    )

    batt_cfg = BatteryConfig.from_cfg()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naïve Dispatch (Baseline)")
        naive = NaiveDispatch(batt_cfg)
        naive_result = naive.run(renewable, demand, price)
        st.metric("Curtailment", f"{naive_result.total_curtailment_mwh:.1f} MWh")
        st.metric("Revenue", f"${naive_result.total_revenue:,.0f}")

    with col2:
        st.subheader("Forecast-Aware Dispatch (AI)")
        smart = ForecastDispatch(batt_cfg)
        smart_result = smart.run(
            renewable,
            renewable * 0.85,
            renewable * 1.15,
            demand,
            price,
        )
        st.metric("Curtailment", f"{smart_result.total_curtailment_mwh:.1f} MWh")
        st.metric("Revenue", f"${smart_result.total_revenue:,.0f}")

    st.divider()
    st.subheader("Battery Health Tracking (Post-Simulation)")
    st.markdown("Tracks hardware degradation caused by calendar aging and cycling stress.")

    colA, colB, colC = st.columns(3)
    colA.metric("Starting Capacity", f"{batt_cfg.capacity_mwh:.2f} MWh")
    colB.metric("Ending Capacity (Smart)", f"{smart.battery.effective_capacity:.4f} MWh")

    capacity_loss = batt_cfg.capacity_mwh - smart.battery.effective_capacity
    colC.metric(
        "Capacity Loss",
        f"{capacity_loss:.6f} MWh",
        delta=f"-{capacity_loss:.6f} MWh",
        delta_color="inverse",
    )
    st.divider()

    # SoC chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=[d.battery_soc for d in naive_result.decisions],
            name="Naïve SoC",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=[d.battery_soc for d in smart_result.decisions],
            name="Smart SoC",
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="Battery State of Charge",
        yaxis_title="SoC",
        xaxis_title="Hour",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Curtailment comparison
    fig2 = go.Figure(
        data=[
            go.Bar(
                name="Naïve",
                x=list(range(24)),
                y=[d.curtailed_mw for d in naive_result.decisions],
            ),
            go.Bar(
                name="Smart",
                x=list(range(24)),
                y=[d.curtailed_mw for d in smart_result.decisions],
            ),
        ]
    )
    fig2.update_layout(title="Hourly Curtailment (MW)", barmode="group", height=350)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE: Metrics
# ══════════════════════════════════════════════════════════════════════
elif page == "Metrics":
    st.title("Model Evaluation Metrics")

    baseline = fit_baseline(data.train.y, data.train.y, data.val.y, data.val.y)
    test_pred = moving_average_forecast(
        data.test.y,
        data.test.y.shape[1],
        min(baseline.best_window, data.test.y.shape[1]),
    )
    actual = data.test.y

    metrics = compute_all_metrics(actual, test_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics['mae']:.4f}")
    col2.metric("RMSE", f"{metrics['rmse']:.4f}")
    col3.metric("MAPE", f"{metrics['mape']:.2f}%")
    col4.metric("R²", f"{metrics['r_squared']:.4f}")

    st.subheader("Target: ≥15% Curtailment Reduction")
    target_pct = CFG["evaluation"]["target_curtailment_reduction"] * 100
    # Quick dispatch test
    horizon = 24
    rng = np.random.default_rng(42)
    ren = 80 + 40 * np.sin(2 * np.pi * np.arange(horizon) / 24) + rng.normal(0, 10, horizon)
    dem = 400 + 150 * np.sin(2 * np.pi * np.arange(horizon) / 24) + rng.normal(0, 20, horizon)
    pri = np.full(horizon, 60.0)

    batt_cfg = BatteryConfig.from_cfg()
    naive_res = NaiveDispatch(batt_cfg).run(np.clip(ren, 0, 200), dem, pri)
    smart_res = ForecastDispatch(batt_cfg).run(
        np.clip(ren, 0, 200),
        np.clip(ren * 0.85, 0, 200),
        np.clip(ren * 1.15, 0, 200),
        dem,
        pri,
    )
    curt_red = curtailment_reduction(
        naive_res.total_curtailment_mwh,
        smart_res.total_curtailment_mwh,
    )

    if curt_red >= target_pct:
        st.success(f"Curtailment reduced by **{curt_red:.1f}%** — TARGET MET!")
    else:
        st.warning(f"Curtailment reduced by **{curt_red:.1f}%** — below {target_pct:.0f}% target.")

    st.progress(min(curt_red / target_pct, 1.0))
