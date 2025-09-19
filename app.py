# app.py
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
import logging

from supabase_logger import SupabaseLogger
from asset_objects import assets_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    verbose_training = st.checkbox("Verbose training output", value=False)

    st.markdown("---")
    st.header("Backtest / Exec thresholds")
    p_fast = st.slider("p_fast", 0.0, 1.0, 0.70)
    p_slow = st.slider("p_slow", 0.0, 1.0, 0.55)
    p_deep = st.slider("p_deep", 0.0, 1.0, 0.45)

    st.markdown("---")
    st.header("Misc")
    force_run = st.checkbox("Force run pipeline (ignore HealthGauge gating)", value=False)
    allow_single_class_train = st.checkbox("Allow training with single-class labels (dangerous)", value=False)
    run = st.button("Run demo pipeline")

if not run:
    st.stop()

# Run pipeline
st.info(f"Fetching price data for {symbol}…")
try:
    bars = fetch_price(
        symbol,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval=interval,
    )
except Exception as e:
    st.error(f"Error fetching price: {e}")
    st.stop()

if bars is None or bars.empty:
    st.error("No price data returned. Check symbol and time range.")
    st.stop()
# ensure index is timezone-aware and unique
bars.index = pd.to_datetime(bars.index, utc=True)
if bars.index.duplicated().any():
    logger.warning("Duplicate timestamps present in raw bars — keeping first occurrence.")
    bars = bars[~bars.index.duplicated(keep="first")]

st.success(f"Fetched {len(bars)} bars.")

# COT + HealthGauge
st.info("Fetching COT & computing HealthGauge…")
try:
    client = init_socrata_client()
    cot = fetch_cot(
        client,
        start=(start_date - timedelta(days=365)).isoformat(),
        end=end_date.isoformat(),
    )
except Exception as e:
    st.warning(f"COT fetch failed: {e} — proceeding with empty COT.")
    cot = pd.DataFrame()

# Build daily bars and dedupe
daily_bars = (
    bars.resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
)
if daily_bars.index.duplicated().any():
    logger.warning("Duplicate daily indexes after resample — keeping last occurrence.")
    daily_bars = daily_bars[~daily_bars.index.duplicated(keep="last")]

# compute health gauge
try:
    health_df = calculate_health_gauge(cot, daily_bars)
except Exception as e:
    st.error(f"HealthGauge computation failed: {e}")
    st.stop()

if health_df.empty:
    st.warning("HealthGauge returned empty frame — proceeding only if force_run is True.")
    if not force_run:
        st.stop()

# dedupe health_df
if health_df.index.duplicated().any():
    health_df = health_df[~health_df.index.duplicated(keep="last")]

latest_health = float(health_df["health_gauge"].iloc[-1])
st.metric("Latest HealthGauge", f"{latest_health:.4f}")
st.line_chart(health_df["health_gauge"].rename("HealthGauge"))

buy_allowed = latest_health >= buy_threshold
sell_allowed = latest_health <= sell_threshold
st.write(f"Buy allowed: {buy_allowed} — Sell allowed: {sell_allowed}")

if not (buy_allowed or sell_allowed or force_run):
    st.warning("HealthGauge not in buy/sell band and force_run is False. Pipeline halted.")
    st.stop()

# Compute rvol and generate candidates
st.info("Computing RVol and generating candidate events…")
bars_rvol = compute_rvol(bars, window=asset_obj.rvol_lookback)

try:
    candidates = generate_candidates_and_labels(
        bars_rvol,
        lookback=64,
        k_tp=3.0,  # training uses TP=3R default per your request
        k_sl=1.0,
        atr_window=asset_obj.atr_lookback,
        max_bars=60,
        rvol_threshold=1.5,
        direction="long",
    )
except Exception as e:
    st.error(f"Error generating candidates: {e}")
    st.stop()

if candidates is None or candidates.empty:
    st.error("No candidates generated. Try different date range or interval.")
    st.stop()

# Ensure candidate_time is datetime and unique index for training convenience
candidates = candidates.copy()
if "candidate_time" in candidates.columns:
    candidates["candidate_time"] = pd.to_datetime(candidates["candidate_time"], utc=True)
    candidates = candidates.sort_values("candidate_time").reset_index(drop=True)
    # set index to candidate_time for easier alignment with predict/probs
    candidates.set_index("candidate_time", inplace=True)
else:
    # if candidate_time missing, rely on index; ensure datetime index
    candidates.index = pd.to_datetime(candidates.index, utc=True)

st.write("Candidate labeling diagnostics")
st.write(f"Total candidates produced: {len(candidates)}")
if "label" in candidates.columns:
    st.write("Label value counts:", candidates["label"].value_counts(dropna=False).to_dict())
else:
    st.warning("No 'label' column in candidates.")

# Merge daily COT-derived features into candidates (align by date)
candidates = candidates.reset_index().rename(columns={"index": "candidate_time"})
candidates["candidate_date"] = pd.to_datetime(candidates["candidate_time"]).dt.normalize()

cot_feats = health_df[["noncomm_net_chg", "comm_net_chg"]].copy()
cot_feats = cot_feats.reset_index().rename(columns={"index": "candidate_date"})
if "candidate_date" in cot_feats.columns and "noncomm_net_chg" in cot_feats.columns:
    candidates = candidates.merge(cot_feats, on="candidate_date", how="left")
else:
    candidates["noncomm_net_chg"] = 0.0
    candidates["comm_net_chg"] = 0.0

candidates = candidates.drop(columns=["candidate_date"]).fillna({"noncomm_net_chg": 0.0, "comm_net_chg": 0.0})

# Prepare features for training
feat_cols = [
    "tick_rate", "uptick_ratio", "buy_vol_ratio",
    "micro_range", "rvol_micro",
    "noncomm_net_chg", "comm_net_chg",
]

# Ensure required columns exist and are numeric
for c in feat_cols:
    if c not in candidates.columns:
        candidates[c] = 0.0
candidates[feat_cols] = candidates[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Clean labels
if "label" not in candidates.columns:
    st.error("Candidates missing 'label' column — cannot train.")
    st.stop()

candidates["label"] = pd.to_numeric(candidates["label"], errors="coerce")
clean = candidates.dropna(subset=["label"]).copy()
clean = clean[clean["label"].isin([0, 1])].reset_index(drop=True)

st.write("Post-clean diagnostics")
st.write(f"Rows before cleaning: {len(candidates)}, after cleaning: {len(clean)}")
if not clean.empty:
    st.write("Label counts:", clean["label"].value_counts().to_dict())
    st.write("NaNs in features:", clean[feat_cols].isna().sum().to_dict())

# Handle single-class label situation
unique_labels = clean["label"].unique()
if len(unique_labels) < 2:
    if allow_single_class_train:
        st.warning(f"Only one class present in labels: {unique_labels}. Proceeding because allow_single_class_train=True.")
    else:
        st.error(f"Only one class present in labels: {unique_labels}. Cannot train. Toggle 'Allow training with single-class labels' to override.")
        st.stop()

# Train model
st.info("Training XGBoost confirm model…")
try:
    model, feature_list, metrics = train_xgb_confirm(
        clean,
        feature_cols=feat_cols,
        label_col="label",
        num_boost_round=int(num_boost),
        early_stopping_rounds=int(early_stop) if int(early_stop) > 0 else None,
        test_size=float(test_size),
        random_state=42,
        verbose=verbose_training,
    )
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

st.write("Training metrics:", metrics)

# Predict and backtest
st.info("Predicting confirm probabilities & running backtest…")
try:
    probs = predict_confirm_prob(model, clean, feature_list)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# align probs to candidates index used by simulate_limits
clean_for_bt = clean.set_index(pd.to_datetime(clean["candidate_time"]))
probs.index = clean_for_bt.index

trades = simulate_limits(bars, clean_for_bt, probs, p_fast=p_fast, p_slow=p_slow, p_deep=p_deep)

# Metrics summary
if not trades.empty:
    trades["ret"] = pd.to_numeric(trades["ret"], errors="coerce").fillna(0.0)
    trades["size"] = pd.to_numeric(trades["size"], errors="coerce").fillna(0.0)
    trades["pnl"] = trades["size"] * trades["ret"]

    num_trades = len(trades)
    total_pnl = trades["pnl"].sum()
    avg_ret = trades["ret"].mean() if num_trades > 0 else 0.0
    win_rate = (trades["ret"] > 0).mean() if num_trades > 0 else 0.0

    st.metric("Num trades", f"{num_trades}")
    st.metric("Total PnL", f"{total_pnl:.6f}")
    st.metric("Average Ret.", f"{avg_ret:.6f}")
    st.metric("Win rate", f"{win_rate:.2%}")

    st.dataframe(trades.head())
    pnl_curve = trades.groupby("candidate_time")["pnl"].sum().cumsum()
    st.line_chart(pnl_curve.rename("Cumulative PnL"))
else:
    st.warning("No trades simulated.")
    num_trades = 0; total_pnl = avg_ret = win_rate = 0.0

st.success("Demo complete.")

# Save model
st.subheader("Save Model")
model_name_input = st.text_input("Enter model name", value=f"confirm_model_{symbol.replace('=','_')}")
if st.button("Save model as .pt"):
    torch.save({"model": model, "features": feature_list}, f"{model_name_input}.pt")
    st.success(f"Saved model to {model_name_input}.pt")

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
        "feature_cols": feature_list,
        "model_file": f"{model_name_input}.pt",
        "training_params": {"num_boost_round": int(num_boost), "early_stopping_rounds": int(early_stop), "test_size": float(test_size)},
        "health_thresholds": {"buy_threshold": float(buy_threshold), "sell_threshold": float(sell_threshold)},
        "p_fast": float(p_fast), "p_slow": float(p_slow), "p_deep": float(p_deep),
    }
    backtest_metrics = {**metrics,
                        "num_trades": int(num_trades),
                        "total_pnl": float(total_pnl),
                        "avg_ret": float(avg_ret),
                        "win_rate": float(win_rate),
                        "latest_health": float(latest_health)}
    supa = SupabaseLogger()
    supa.log_run(metrics=backtest_metrics, metadata=metadata,
                 trades=[{**r, "run_id": run_id} for r in trades.to_dict("records")] if not trades.empty else [])
    st.success(f"Logged run {run_id} ✔")