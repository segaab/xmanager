# app.py (full file) — regenerated (preserves all logic; small updates noted in comments)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from fetch_data import fetch_price, init_socrata_client, fetch_cot
from features import compute_rvol, calculate_health_gauge
from labeling import generate_candidates_and_labels
from model import train_xgb_confirm, predict_confirm_prob
from backtest import simulate_limits
import matplotlib.pyplot as plt
import uuid
import torch

# new import
from supabase_logger import SupabaseLogger
from asset_objects import assets_list

st.set_page_config(layout="wide", page_title="Entry Triangulation Demo")

st.title("Entry-Range Triangulation Demo (HealthGauge → Entry → Confirm)")

# Sidebar controls
with st.sidebar:
    st.header("Data / Run Controls")
    selected_asset = st.selectbox("Select asset", options=[a.name for a in assets_list])
    asset_obj = next(a for a in assets_list if a.name == selected_asset)
    symbol = asset_obj.symbol

    start_date = st.date_input("Start date", value=date.today() - timedelta(days=180))
    end_date = st.date_input("End date", value=date.today())
    interval = st.selectbox("Bar interval", options=["1m", "5m", "1h", "1d"], index=1)

    st.markdown("---")
    st.header("HealthGauge Thresholds")
    buy_threshold = st.slider("Buy threshold (HealthGauge ≥)", 0.0, 1.0, 0.60, 0.01)
    sell_threshold = st.slider("Sell threshold (HealthGauge ≤)", 0.0, 1.0, 0.40, 0.01)

    st.markdown("---")
    st.header("Training Controls (XGBoost)")
    num_boost = st.number_input("Boosting rounds", value=500, step=50)
    early_stop = st.number_input("Early stopping rounds", value=20, step=5)
    test_size = st.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)

    st.markdown("---")
    st.header("Backtest / Exec thresholds")
    p_fast = st.slider("p_fast", 0.0, 1.0, 0.70)
    p_slow = st.slider("p_slow", 0.0, 1.0, 0.55)
    p_deep = st.slider("p_deep", 0.0, 1.0, 0.45)

    st.markdown("---")
    force_run = st.checkbox("Force run pipeline", value=False)
    run = st.button("Run demo pipeline")

if run:
    # --- Small client-side warnings for intraday limit violations (individualized rules)
    days_range = (end_date - start_date).days
    if interval == "1m" and days_range > 30:
        st.warning("Interval 1m has a 30-day max window per Yahoo. The request will be batched; expect longer fetch time and possible missing chunks.")
    if interval in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"} and days_range > 60:
        st.warning("Selected intraday interval has a ~60-day max window per chunk. The request will be batched; expect longer fetch time.")

    st.info(f"Fetching price data for {symbol} (may be batched for long ranges)…")
    try:
        # Pass max_years=15 so fetch_price can internally cap/batch up to 15 years
        bars = fetch_price(
            symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval=interval,
            max_years=15,
        )
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        st.stop()

    if bars.empty:
        st.error("No price data returned. Check symbol and time range.")
        st.stop()
    st.success(f"Fetched {len(bars)} bars.")

    st.info("Fetching COT and computing HealthGauge…")
    try:
        client = init_socrata_client()
        # request up to 15 years of COT (fetch_cot will batch/thread internally)
        cot = fetch_cot(
            client,
            start=(start_date - timedelta(days=365 * 15)).isoformat(),
            end=end_date.isoformat(),
            cot_name=asset_obj.cot_name,
            max_years=15,
        )
    except Exception as e:
        st.warning(f"COT fetch failed: {e}. Proceeding with empty COT.")
        cot = pd.DataFrame()

    # Build daily bars and compute gauge
    daily_bars = bars.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    health_df = calculate_health_gauge(cot, daily_bars)
    if health_df.empty:
        st.warning("HealthGauge computation returned empty dataframe.")
        st.stop()

    latest_health = float(health_df['health_gauge'].iloc[-1])
    st.metric("Latest HealthGauge", f"{latest_health:.4f}")
    st.line_chart(health_df[['health_gauge']].rename(columns={'health_gauge': 'HealthGauge'}))

    buy_allowed = latest_health >= buy_threshold
    sell_allowed = latest_health <= sell_threshold
    st.write(f"Buy allowed: {buy_allowed}, Sell allowed: {sell_allowed}")

    if not (buy_allowed or sell_allowed or force_run):
        st.warning("HealthGauge not in buy/sell band. Pipeline halted.")
        st.stop()

    st.info("Computing RVol and generating candidate events…")
    # use asset-configured lookback
    bars_rvol = compute_rvol(bars, window=asset_obj.rvol_lookback)

    try:
        candidates = generate_candidates_and_labels(
            bars_rvol,
            lookback=64,
            k_tp=2.0,
            k_sl=1.0,
            atr_window=asset_obj.atr_lookback,
            max_bars=60,
        )
    except Exception as e:
        st.error(f"Error generating candidates: {e}")
        st.stop()

    st.write("Candidates generated:", len(candidates))
    if len(candidates) == 0:
        st.warning("No candidates. Try a different date range or interval.")
        st.stop()
    st.dataframe(candidates.head())

    st.info("Training XGBoost confirm model…")
    feat_cols = ['tick_rate', 'uptick_ratio', 'buy_vol_ratio', 'micro_range', 'rvol_micro']
    missing = [c for c in feat_cols if c not in candidates.columns]
    if missing:
        st.error(f"Missing candidate features: {missing}")
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

    if not trades.empty:
        trades['ret'] = pd.to_numeric(trades.get('ret', 0.0), errors='coerce').fillna(0.0)
        trades['size'] = pd.to_numeric(trades.get('size', 0.0), errors='coerce').fillna(0.0)
        trades['pnl'] = trades['size'] * trades['ret']

        num_trades = len(trades)
        total_pnl = trades['pnl'].sum()
        avg_ret = trades['ret'].mean()
        win_rate = (trades['ret'] > 0).sum() / num_trades

        st.metric("Num trades", f"{num_trades}")
        st.metric("Total PnL", f"{total_pnl:.6f}")
        st.metric("Average return", f"{avg_ret:.6f}")
        st.metric("Win rate", f"{win_rate:.2%}")

        st.dataframe(trades.head())
        pnl_series = trades.groupby('candidate_time')['pnl'].sum().cumsum()
        fig, ax = plt.subplots()
        pnl_series.plot(ax=ax)
        ax.set_title("Cumulative PnL")
        st.pyplot(fig)
    else:
        st.warning("No trades simulated.")

    st.success("Demo complete.")

    # Save model as .pt
    st.subheader("Save Model")
    model_name_input = st.text_input("Enter model name", value=f"confirm_model_{symbol.replace('=','_')}")
    if st.button("Save model as .pt"):
        if 'model_booster' in locals() and 'feature_list' in locals():
            model_fname = f"{model_name_input}.pt"
            # Use torch.save to persist the model dict (compatible with earlier .pt requirement)
            torch.save({'model': model_booster, 'features': feature_list}, model_fname)
            st.success(f"Saved model to {model_fname}")
        else:
            st.error("No trained model found. Run training first.")

    # Supabase logging
    st.subheader("Logging")
    if st.button("Save logs to Supabase"):
        run_id = str(uuid.uuid4())
        metadata = {
            "run_id": run_id,
            "symbol": symbol,
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