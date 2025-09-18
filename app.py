# app.py (full file)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from fetch_data import fetch_price, init_socrata_client, fetch_cot
from features import compute_rvol, calculate_health_gauge
from labeling import generate_candidates_and_labels
from model import train_xgb_confirm, predict_confirm_prob
from backtest import simulate_limits
import joblib
import matplotlib.pyplot as plt
import os
import json
import uuid

# new import
from supabase_logger import SupabaseLogger
from assets_list import assets_list  # new modal for asset objects

st.set_page_config(layout="wide", page_title="Entry Triangulation Demo")

st.title("Entry-Range Triangulation Demo (HealthGauge → Entry → Confirm)")

# Sidebar controls: data selection + training + thresholds
with st.sidebar:
    st.header("Asset Selection")
    asset_names = [a.name for a in assets_list]
    selected_asset_name = st.selectbox("Select asset", options=asset_names, index=0)
    asset = next(a for a in assets_list if a.name == selected_asset_name)

    st.markdown("---")
    st.header("Data / Run Controls")
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=180))
    end_date = st.date_input("End date", value=date.today())
    interval = st.selectbox("Bar interval", options=["1m", "5m", "1h", "1d"], index=1)

    st.markdown("---")
    st.header("HealthGauge Thresholds")
    buy_threshold = st.slider("Buy threshold (HealthGauge ≥)", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
    sell_threshold = st.slider("Sell threshold (HealthGauge ≤)", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
    st.caption("HealthGauge is normalized to [0,1]. If current gauge ≥ buy_threshold, we allow long opportunities. If ≤ sell_threshold, we allow short opportunities. Use force-run to override.")

    st.markdown("---")
    st.header("Training Controls (XGBoost)")
    num_boost = st.number_input("Boosting rounds (num_boost_round)", value=500, step=50)
    early_stop = st.number_input("Early stopping rounds", value=20, step=5)
    test_size = st.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.20, step=0.05)
    st.caption("These parameters are passed directly to train_xgb_confirm()")

    st.markdown("---")
    st.header("Backtest / Exec thresholds")
    p_fast = st.slider("p_fast (ideal post threshold)", min_value=0.0, max_value=1.0, value=0.70)
    p_slow = st.slider("p_slow (shallow post threshold)", min_value=0.0, max_value=1.0, value=0.55)
    p_deep = st.slider("p_deep (deep post threshold)", min_value=0.0, max_value=1.0, value=0.45)

    st.markdown("---")
    force_run = st.checkbox("Force run pipeline (ignore HealthGauge gating)", value=False)
    run = st.button("Run demo pipeline")

if run:
    st.info(f"Fetching price data for {asset.symbol} (yahooquery)… this may take a few seconds.")
    try:
        bars = fetch_price(asset.symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        st.stop()

    if bars.empty:
        st.error("No price data returned. Check symbol and time range.")
        st.stop()
    st.success(f"Fetched {len(bars)} bars ({interval}).")

    st.info(f"Fetching COT data for {asset.cot_name} (socrata)…")
    try:
        client = init_socrata_client()
        cot = fetch_cot(client,
                        start=(start_date - timedelta(days=365)).isoformat(),
                        end=end_date.isoformat(),
                        cot_name=asset.cot_name)
    except Exception as e:
        st.warning(f"Error fetching COT or Socrata client init: {e}. Proceeding with empty COT (health gauge will be based on volume proxies).")
        cot = pd.DataFrame()

    # Build daily bars for health gauge alignment
    daily_bars = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    health_df = calculate_health_gauge(cot, daily_bars)
    if health_df.empty:
        st.warning("HealthGauge computation returned empty dataframe.")
        st.stop()

    latest_health = float(health_df['health_gauge'].iloc[-1])
    st.metric("Latest HealthGauge", f"{latest_health:.4f}")
    st.line_chart(health_df[['health_gauge']].rename(columns={'health_gauge':'HealthGauge'}))

    # Decision gating
    buy_allowed = latest_health >= buy_threshold
    sell_allowed = latest_health <= sell_threshold

    st.write(f"Buy allowed: {buy_allowed} (threshold {buy_threshold})")
    st.write(f"Sell allowed: {sell_allowed} (threshold {sell_threshold})")

    if not (buy_allowed or sell_allowed or force_run):
        st.warning("HealthGauge not in buy/sell band and force_run is False. Pipeline halted. Toggle 'Force run pipeline' to override.")
        st.stop()

    st.info("Computing RVol and generating candidate events (labels)…")
    bars_rvol = compute_rvol(bars, window=asset.rvol_lookback)

    try:
        candidates = generate_candidates_and_labels(bars_rvol, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=asset.atr_lookback, max_bars=60)
    except Exception as e:
        st.error(f"Error generating candidates: {e}")
        st.stop()

    st.write("Candidates generated:", len(candidates))
    if len(candidates) == 0:
        st.warning("No candidates generated for this dataset/timeframe. Try extending date range or using a different interval.")
        st.stop()
    st.dataframe(candidates.head())

    st.info("Training XGBoost confirm model (features approximated from bars)…")
    feat_cols = ['tick_rate','uptick_ratio','buy_vol_ratio','micro_range','rvol_micro']
    missing = [c for c in feat_cols if c not in candidates.columns]
    if missing:
        st.error(f"Missing required candidate features: {missing}")
        st.stop()

    try:
        model_booster, feature_list, metrics = train_xgb_confirm(
            candidates,
            feature_cols=feat_cols,
            label_col='label',
            save_path=None,
            num_boost_round=int(num_boost),
            early_stopping_rounds=int(early_stop),
            test_size=float(test_size),
            random_state=42,
            verbose=False,
        )
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    st.write("Training metrics:", metrics)

    st.info("Predicting confirm probabilities and running backtest…")
    try:
        probs = predict_confirm_prob(model_booster, candidates, feature_list)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    trades = simulate_limits(bars, candidates, probs, p_fast=p_fast, p_slow=p_slow, p_deep=p_deep)
    st.write("Simulated trades:", len(trades))

    # Compute and display trade metrics
    if not trades.empty:
        trades['ret'] = pd.to_numeric(trades.get('ret', 0.0), errors='coerce').fillna(0.0)
        trades['size'] = pd.to_numeric(trades.get('size', 0.0), errors='coerce').fillna(0.0)
        trades['pnl'] = trades['size'] * trades['ret']

        num_trades = len(trades)
        total_pnl = trades['pnl'].sum()
        avg_ret = trades['ret'].mean()
        median_ret = trades['ret'].median()
        std_ret = trades['ret'].std(ddof=0) if num_trades > 1 else 0.0
        win_rate = (trades['ret'] > 0).sum() / num_trades

        st.metric("Num trades (simulated)", f"{num_trades}")
        st.metric("Total PnL (simulated)", f"{total_pnl:.6f}")
        st.metric("Average return / filled trade", f"{avg_ret:.6f}")
        st.metric("Win rate", f"{win_rate:.2%}")

        st.dataframe(trades.head())

        pnl_series = trades.groupby('candidate_time')['pnl'].sum().cumsum()
        fig, ax = plt.subplots()
        pnl_series.plot(ax=ax)
        ax.set_title("Cumulative PnL (simulated)")
        st.pyplot(fig)
    else:
        st.warning("No trades simulated — model may be too strict or no fills occurred.")

    st.success("Demo complete.")

    # Save model & optionally log to Supabase
    if st.button("Save model & feature list"):
        model_fname = f'confirm_model_{asset.symbol.replace("=","_")}_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.pkl'
        joblib.dump({'model': model_booster, 'features': feature_list}, model_fname)
        st.write(f"Saved model to {model_fname}")

    # Log to Supabase
    st.markdown("---")
    st.subheader("Logging")
    st.write("You can log this run's summary metrics and per-trade records to Supabase for later analysis.")
    if st.button("Save logs to Supabase"):
        run_id = str(uuid.uuid4())
        metadata = {
            "run_id": run_id,
            "asset_name": asset.name,
            "symbol": asset.symbol,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "interval": interval,
            "feature_cols": feat_cols,
            "model_file": model_fname if 'model_fname' in locals() else None,
            "training_params": {"num_boost_round": int(num_boost), "early_stopping_rounds": int(early_stop), "test_size": float(test_size)},
            "health_thresholds": {"buy_threshold": float(buy_threshold), "sell_threshold": float(sell_threshold)},
            "p_fast": float(p_fast), "p_slow": float(p_slow), "p_deep": float(p_deep),
        }

        backtest_metrics = {
            "num_trades": int(num_trades) if not trades.empty else 0,
            "total_pnl": float(total_pnl) if not trades.empty else 0.0,
            "avg_ret": float(avg_ret) if not trades.empty else 0.0,
            "median_ret": float(median_ret) if not trades.empty else 0.0,
            "std_ret": float(std_ret) if not trades.empty else 0.0,
            "win_rate": float(win_rate) if not trades.empty else 0.0,
            "latest_health": float(latest_health),
        }
        combined_metrics = dict(metrics)
        combined_metrics.update(backtest_metrics)

        trade_list = []
        if not trades.empty:
            for r in trades.to_dict(orient="records"):
                trade_list.append({
                    "candidate_time": str(r.get("candidate_time")),
                    "layer": r.get("layer"),
                    "size": float(r.get("size") or 0.0),
                    "entry_price": float(r.get("entry_price") or 0.0),
                    "filled_at": str(r.get("filled_at")) if r.get("filled_at") is not None else None,
                    "ret": float(r.get("ret") or 0.0),
                    "pnl": float(r.get("pnl") or 0.0),
                })

        try:
            supa = SupabaseLogger()
            run_id_returned = supa.log_run(metrics=combined_metrics, metadata=metadata, trades=trade_list)
            st.success(f"Logged run to Supabase with run_id: {run_id_returned}")
        except Exception as e:
            st.error(f"Failed to log to Supabase: {e}")