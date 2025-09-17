# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from fetch_data import fetch_price, init_socrata_client, fetch_cot
from features import compute_rvol, calculate_health_gauge
from labeling import generate_candidates_and_labels
from model import train_xgb_confirm, predict_confirm_prob
from backtest import simulate_limits
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Entry Triangulation Demo")

st.title("Entry-Range Triangulation Demo (HealthGauge → Entry → Confirm)")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Symbol (Yahoo)", value="GC=F")  # Gold futures example
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=180))
    end_date = st.date_input("End date", value=date.today())
    interval = st.selectbox("Bar interval", options=["1m","5m","1h","1d"], index=1)
    run = st.button("Run demo pipeline")

if run:
    st.info("Fetching price data (yahooquery)… this may take a few seconds.")
    bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    if bars.empty:
        st.error("No price data returned. Check symbol and time range.")
        st.stop()
    st.success(f"Fetched {len(bars)} bars.")

    st.info("Fetching COT (socrata) and computing HealthGauge…")
    client = init_socrata_client()
    cot = fetch_cot(client, start=(start_date - timedelta(days=365)).isoformat(), end=end_date.isoformat())
    health_df = calculate_health_gauge(cot, bars.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna())
    st.line_chart(health_df[['health_gauge']].rename(columns={'health_gauge':'HealthGauge'}))

    st.info("Computing RVol and generating candidate events (labels)…")
    bars_rvol = compute_rvol(bars, window=20)
    # For candidate generation we need reasonably many rows -> use minute/5m data; if interval too coarse adjust
    candidates = generate_candidates_and_labels(bars_rvol, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=20, max_bars=60)
    st.write("Candidates:", len(candidates))
    st.dataframe(candidates.head())

    st.info("Training XGBoost confirm model (features from minute bars approximating ticks)…")
    model, features, metrics = train_xgb_confirm(candidates, feature_cols=['tick_rate','uptick_ratio','buy_vol_ratio','micro_range','rvol_micro'])
    st.write("Training metrics:", metrics)

    st.info("Predicting confirm probabilities and running backtest…")
    probs = predict_confirm_prob(model, candidates, features)
    trades = simulate_limits(bars, candidates, probs, p_fast=0.7, p_slow=0.55, p_deep=0.45)
    st.write("Simulated trades:", len(trades))
    if not trades.empty:
        st.dataframe(trades.head())
        # basic performance
        avg_ret = trades['ret'].astype(float).mean()
        st.metric("Average realized return per filled trade", f"{avg_ret:.4f}")
        # cumulative PnL by simple portfolio (size as weight)
        trades['pnl'] = trades['size'] * trades['ret'].astype(float)
        pnl_series = trades.groupby('candidate_time')['pnl'].sum().cumsum()
        fig, ax = plt.subplots()
        pnl_series.plot(ax=ax)
        ax.set_title("Cumulative PnL (simulated)")
        st.pyplot(fig)
    else:
        st.warning("No trades simulated — model may be too strict or no fills occurred.")

    st.success("Demo complete. Export model to disk?")
    if st.button("Save model & features"):
        joblib.dump({'model':model, 'features':features}, f'confirm_model_{symbol.replace("=","_")}.pkl')
        st.write("Saved confirm_model file.")