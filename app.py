# app.py – Entry-Range Triangulation Demo (patched)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from fetch_data import fetch_price, init_socrata_client, fetch_cot
from features import compute_rvol, calculate_health_gauge
from labeling import generate_candidates_and_labels
from model import train_xgb_confirm, predict_confirm_prob
from backtest import simulate_limits
import matplotlib.pyplot as plt
import uuid, torch
from supabase_logger import SupabaseLogger
from asset_objects import assets_list

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Entry Triangulation Demo")
st.title("Entry-Range Triangulation Demo (HealthGauge → Entry → Confirm)")

# ────────── sidebar ──────────
with st.sidebar:
    st.header("Data / Run Controls")
    selected_asset = st.selectbox("Select asset", options=[a.name for a in assets_list])
    asset_obj = next(a for a in assets_list if a.name == selected_asset)
    symbol = asset_obj.symbol

    start_date = st.date_input("Start date", value=date.today() - timedelta(days=180))
    end_date   = st.date_input("End date",   value=date.today())
    interval   = st.selectbox("Bar interval", ["1m", "5m", "1h", "1d"], index=1)

    st.markdown("---")
    st.header("HealthGauge Thresholds")
    buy_threshold  = st.slider("Buy threshold  ≥", 0.0, 1.0, 0.60, 0.01)
    sell_threshold = st.slider("Sell threshold ≤", 0.0, 1.0, 0.40, 0.01)

    st.markdown("---")
    st.header("Training Controls (XGBoost)")
    num_boost  = st.number_input("Boosting rounds", value=500, step=50)
    early_stop = st.number_input("Early stopping rounds", value=20, step=5)
    test_size  = st.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)

    st.markdown("---")
    st.header("Backtest / Exec thresholds")
    p_fast = st.slider("p_fast", 0.0, 1.0, 0.70)
    p_slow = st.slider("p_slow", 0.0, 1.0, 0.55)
    p_deep = st.slider("p_deep", 0.0, 1.0, 0.45)

    st.markdown("---")
    force_run = st.checkbox("Force run pipeline", value=False)
    run       = st.button("Run demo pipeline")

# ────────── pipeline ──────────
if run:
    # 1) PRICE -----------------------------------------------------------------
    st.info(f"Fetching price data for {symbol}…")
    bars = fetch_price(symbol, start=start_date.isoformat(),
                       end=end_date.isoformat(), interval=interval)
    if bars.empty:
        st.error("No price data returned.")
        st.stop()
    st.success(f"Fetched {len(bars)} bars.")

    # 2) COT + HEALTH ----------------------------------------------------------
    st.info("Fetching COT & computing HealthGauge…")
    try:
        client = init_socrata_client()
        cot    = fetch_cot(client,
                           start=(start_date - timedelta(days=365)).isoformat(),
                           end=end_date.isoformat())
    except Exception as e:
        st.warning(f"COT fetch failed: {e}. Continuing with empty COT.")
        cot = pd.DataFrame()

    # Daily bars (ensure unique index)
    daily_bars = (
        bars.resample("1D")
            .agg({"open": "first",
                  "high": "max",
                  "low":  "min",
                  "close":"last",
                  "volume":"sum"})
            .dropna()
    )
    daily_bars = daily_bars.loc[~daily_bars.index.duplicated(keep="first")]

    # HealthGauge (also unique index)
    health_df = calculate_health_gauge(cot, daily_bars)
    health_df = health_df.loc[~health_df.index.duplicated(keep="last")]

    latest_health = float(health_df["health_gauge"].iloc[-1])
    st.metric("Latest HealthGauge", f"{latest_health:.4f}")
    st.line_chart(health_df["health_gauge"].rename("HealthGauge"))

    buy_allowed  = latest_health >= buy_threshold
    sell_allowed = latest_health <= sell_threshold
    if not (buy_allowed or sell_allowed or force_run):
        st.warning("HealthGauge not in buy/sell band. Pipeline halted.")
        st.stop()

    # 3) RVOL + CANDIDATES -----------------------------------------------------
    st.info("Computing RVol & generating candidate events…")
    bars_rvol = compute_rvol(bars, window=asset_obj.rvol_lookback)
    candidates = generate_candidates_and_labels(
        bars_rvol, lookback=64, k_tp=2.0, k_sl=1.0,
        atr_window=asset_obj.atr_lookback, max_bars=60
    )
    if candidates.empty:
        st.error("No candidates generated.")
        st.stop()

    # inject daily COT net-change features into *intraday* candidates
    candidates = candidates.copy()
    candidates["candidate_date"] = pd.to_datetime(
        candidates["candidate_time"]
    ).dt.normalize()

    cot_feats = (
        health_df[["noncomm_net_chg", "comm_net_chg"]]
        .loc[~health_df.index.duplicated(keep="last")]
        .reset_index(names="candidate_date")
    )

    candidates = (
        candidates.merge(cot_feats, on="candidate_date", how="left")
                  .drop(columns="candidate_date")
                  .fillna({"noncomm_net_chg": 0.0, "comm_net_chg": 0.0})
    )

    # 4) TRAIN CONFIRM MODEL ---------------------------------------------------
    st.info("Training XGBoost confirm model…")
    feat_cols = [
        "tick_rate", "uptick_ratio", "buy_vol_ratio",
        "micro_range", "rvol_micro",
        "noncomm_net_chg", "comm_net_chg"
    ]

    for col in feat_cols + ["label"]:
        if col not in candidates.columns:
            candidates[col] = np.nan
    for col in feat_cols:
        candidates[col] = pd.to_numeric(candidates[col],
                                        errors="coerce").fillna(0)

    clean = candidates.dropna(subset=["label"])
    clean = clean[clean["label"].isin([0, 1])]
    if clean.empty:
        st.error("No valid labeled candidates after cleaning.")
        st.stop()

    model, feature_list, metrics = train_xgb_confirm(
        clean, feature_cols=feat_cols, label_col="label",
        num_boost_round=int(num_boost),
        early_stopping_rounds=int(early_stop),
        test_size=float(test_size),
        random_state=42,
        verbose=False,
    )
    st.write("Training metrics:", metrics)

    # 5) PREDICT & BACKTEST ----------------------------------------------------
    st.info("Predicting confirm probabilities & running backtest…")
    probs  = predict_confirm_prob(model, clean, feature_list)
    trades = simulate_limits(bars, clean, probs,
                             p_fast=p_fast, p_slow=p_slow, p_deep=p_deep)

    # ---- metrics summary -----------------------------------------------------
    if not trades.empty:
        trades["ret"]  = pd.to_numeric(trades["ret"],  errors="coerce").fillna(0)
        trades["size"] = pd.to_numeric(trades["size"], errors="coerce").fillna(0)
        trades["pnl"]  = trades["size"] * trades["ret"]

        num_trades = len(trades)
        total_pnl  = trades["pnl"].sum()
        avg_ret    = trades["ret"].mean()
        win_rate   = (trades["ret"] > 0).mean()

        st.metric("Num trades",   f"{num_trades}")
        st.metric("Total PnL",    f"{total_pnl:.6f}")
        st.metric("Average Ret.", f"{avg_ret:.6f}")
        st.metric("Win rate",     f"{win_rate:.2%}")

        st.dataframe(trades.head())

        pnl_curve = trades.groupby("candidate_time")["pnl"].sum().cumsum()
        st.line_chart(pnl_curve.rename("Cumulative PnL"))
    else:
        st.warning("No trades simulated.")
        num_trades = 0; total_pnl = avg_ret = win_rate = 0.0

    st.success("Demo complete.")

    # 6) SAVE MODEL (.pt) ------------------------------------------------------
    st.subheader("Save Model")
    model_name_input = st.text_input("Enter model name",
                                     value=f"confirm_model_{symbol.replace('=','_')}")
    if st.button("Save model as .pt"):
        torch.save({"model": model, "features": feature_list},
                   f"{model_name_input}.pt")
        st.success(f"Saved model to {model_name_input}.pt")

    # 7) SUPABASE LOGGING ------------------------------------------------------
    st.subheader("Logging")
    if st.button("Save logs to Supabase"):
        run_id = str(uuid.uuid4())
        metadata = {
            "run_id": run_id,
            "symbol": symbol,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "interval": interval,
            "feature_cols": feature_list,
            "model_file": f"{model_name_input}.pt",
            "training_params": {
                "num_boost_round": int(num_boost),
                "early_stopping_rounds": int(early_stop),
                "test_size": float(test_size),
            },
            "health_thresholds": {
                "buy_threshold":  float(buy_threshold),
                "sell_threshold": float(sell_threshold),
            },
            "p_fast": float(p_fast), "p_slow": float(p_slow), "p_deep": float(p_deep),
        }
        backtest_metrics = {
            **metrics,
            "num_trades": int(num_trades),
            "total_pnl":  float(total_pnl),
            "avg_ret":    float(avg_ret),
            "win_rate":   float(win_rate),
            "latest_health": float(latest_health),
        }
        supa = SupabaseLogger()
        supa.log_run(metrics=backtest_metrics, metadata=metadata,
                     trades=[{**r, "run_id": run_id}
                             for r in trades.to_dict("records")]
                     if not trades.empty else [])
        st.success(f"Logged run {run_id} ✔")
