# app.py — Entry-Range Triangulation Demo (full script, chunk 1/3)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import traceback
import uuid
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report

# Internal imports (assume these exist)
from utils.data_fetch import fetch_price, fetch_cot, init_socrata_client
from utils.health import calculate_health_gauge
from utils.candidates import build_or_fetch_candidates, generate_candidates_and_labels, compute_rvol
from utils.model import train_xgb_confirm, predict_confirm_prob, export_model_and_metadata
from utils.backtest import simulate_limits, run_breadth_backtest, summarize_sweep
from utils.supabase_logger import SupabaseLogger
from utils.helpers import df_to_csv_bytes

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Entry-Range Triangulation Demo", layout="wide")
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# ---------------------------
# User Inputs / Config
# ---------------------------
symbol = st.text_input("Symbol", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=30))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)

buy_threshold = st.number_input("Buy threshold (HealthGauge)", min_value=0.0, max_value=1.0, value=0.55)
sell_threshold = st.number_input("Sell threshold (HealthGauge)", min_value=0.0, max_value=1.0, value=0.45)

num_boost = st.number_input("XGBoost num_boost_round", min_value=1, value=50)
early_stop = st.number_input("XGBoost early_stopping_rounds", min_value=1, value=10)
test_size = st.number_input("Test set fraction", min_value=0.0, max_value=1.0, value=0.2)

p_fast = st.number_input("Threshold fast", min_value=0.0, max_value=1.0, value=0.5)
p_slow = st.number_input("Threshold slow", min_value=0.0, max_value=1.0, value=0.55)
p_deep = st.number_input("Threshold deep", min_value=0.0, max_value=1.0, value=0.6)

force_run = st.checkbox("Force run even outside buy/sell band", value=False)
show_confusion = st.checkbox("Show confusion matrix / classification report", value=True)
overlay_entries_on_price = st.checkbox("Overlay entries on price chart", value=True)
include_health_as_feature = st.checkbox("Include HealthGauge as feature", value=True)
save_feature_importance = st.checkbox("Save feature importance on export", value=True)

# Breadth / sweep buttons
run_breadth = st.button("Run breadth modes (Low → Mid → High)")
run_sweep_btn = st.button("Run grid sweep")

# Asset object (example placeholder)
from dataclasses import dataclass
@dataclass
class Asset:
    name: str
    cot_name: str
    symbol: str
    rvol_lookback: int = 20
    atr_lookback: int = 14

asset_obj = Asset(name="Gold", cot_name="GOLD - COMMODITY EXCHANGE INC.", symbol=symbol)
max_bars = 100  # Example

# RR, SL, MPT configs
rr_vals = [1.0, 1.5, 2.0]
sl_ranges = [(0.5, 1.0), (1.0, 2.0)]
mpt_list = [0.5, 0.6, 0.7]
session_modes = ["Low", "Mid", "High"]

# Initialize empty placeholders
bars = pd.DataFrame()
clean = pd.DataFrame()
health_df = pd.DataFrame()
trades = pd.DataFrame()

# ---------------------------
# Main pipeline
# ---------------------------
run = st.button("Run main pipeline")
if run:
    logger.info("Pipeline run initiated by user.")
    st.info(f"Fetching price data for {symbol}…")
    try:
        bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    except Exception as e:
        logger.error("fetch_price failed: %s", e)
        st.error(f"Error fetching price: {e}")
        st.stop()

    if bars is None or bars.empty:
        st.error("No price data returned. Check symbol and time range.")
        st.stop()
    st.success(f"Fetched {len(bars)} bars.")
    logger.info("Fetched %d bars for %s", len(bars), symbol)

    # COT & HealthGauge
    st.info("Fetching COT & computing HealthGauge…")
    try:
        client = init_socrata_client()
        cot = fetch_cot(client, start=(start_date - timedelta(days=365)).isoformat(), end=end_date.isoformat())
    except Exception as e:
        logger.warning("COT fetch failed: %s", e)
        cot = pd.DataFrame()

    # Build daily bars safely
    try:
        daily_bars = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        daily_bars = daily_bars.loc[~daily_bars.index.duplicated(keep="first")]
    except Exception as exc:
        logger.error("Daily bars resampling failed: %s", exc)
        st.error(f"Daily bars preparation failed: {exc}")
        st.stop()

    try:
        health_df = calculate_health_gauge(cot, daily_bars)
    except Exception as exc:
        logger.error("calculate_health_gauge failed: %s", exc)
        st.error(f"HealthGauge computation failed: {exc}")
        st.stop()

    health_df = health_df.loc[~health_df.index.duplicated(keep="last")]
    if health_df.empty:
        st.warning("HealthGauge computation returned empty dataframe.")
        st.stop()

    latest_health = float(health_df['health_gauge'].iloc[-1])
    st.metric("Latest HealthGauge", f"{latest_health:.4f}")
    st.line_chart(health_df[['health_gauge']].rename(columns={'health_gauge':'HealthGauge'}))

    buy_allowed = latest_health >= buy_threshold
    sell_allowed = latest_health <= sell_threshold
    st.write(f"Buy allowed: {buy_allowed}, Sell allowed: {sell_allowed}")
    logger.info("Latest health: %.4f | buy_allowed=%s sell_allowed=%s", latest_health, buy_allowed, sell_allowed)

    if not (buy_allowed or sell_allowed or force_run):
        st.warning("HealthGauge not in buy/sell band. Pipeline halted.")
        st.stop()

    # Build candidates
    st.info("Computing RVol and generating candidate events…")
    try:
        clean = build_or_fetch_candidates(bars, asset_obj, int(max_bars), include_health_as_feature, health_df)
    except Exception as exc:
        logger.error("Candidate generation failed: %s", exc)
        st.error(f"Candidate generation failed: {exc}")
        st.stop()

    if clean is None or clean.empty:
        st.error("No candidates generated. Try a different date range or interval.")
        st.stop()

    st.write("Candidates generated:", len(clean))
    st.dataframe(clean.head())

    # Train confirm model
    st.info("Training XGBoost confirm model…")
    feat_cols = ['tick_rate','uptick_ratio','buy_vol_ratio','micro_range','rvol_micro']
    if include_health_as_feature:
        feat_cols.append('health_gauge')

    try:
        model, feature_list, metrics = train_xgb_confirm(
            clean=clean,
            feature_cols=feat_cols,
            label_col="label",
            num_boost_round=int(num_boost),
            early_stopping_rounds=int(early_stop),
            test_size=float(test_size),
            random_state=42,
            verbose=False,
        )
    except Exception as e:
        logger.error("train_xgb_confirm failed: %s", e)
        st.error(f"Training failed: {e}")
        st.stop()

    st.write("Training metrics (summary):")
    st.write(metrics)
    logger.info("Training completed. Metrics: %s", metrics)

    # Predict & backtest
    st.info("Predicting confirm probabilities on candidate events…")
    try:
        probs = predict_confirm_prob(model, clean, feature_list)
        clean = clean.copy()
        clean['pred_prob'] = probs.reindex(clean.index).fillna(0.0)
        clean['pred_label'] = (clean['pred_prob'] >= p_fast).astype(int)
    except Exception as e:
        logger.error("predict_confirm_prob failed: %s", e)
        st.error(f"Prediction failed: {e}")
        st.stop()

    if show_confusion:
        try:
            y_true = clean['label'].values
            y_pred = clean['pred_label'].values
            cm = confusion_matrix(y_true, y_pred)
            cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            st.subheader("Confusion matrix (threshold={:.2f})".format(p_fast))
            st.write(pd.DataFrame(cm, index=['true_0','true_1'], columns=['pred_0','pred_1']))
            st.subheader("Classification report")
            st.write(pd.DataFrame(cr).transpose())
        except Exception as exc:
            logger.warning("Confusion/classification generation failed: %s", exc)
            st.warning(f"Could not compute confusion/classification report: {exc}")

    st.info("Running simulated fills & backtest…")
    try:
        trades = simulate_limits(bars, clean, probs, p_fast=p_fast, p_slow=p_slow, p_deep=p_deep)
    except Exception as e:
        logger.error("simulate_limits failed: %s", e)
        st.error(f"Backtest simulation failed: {e}")
        st.stop()

    st.write("Simulated trades:", 0 if trades is None else len(trades))
    if trades is None or trades.empty:
        st.warning("No trades simulated.")
        trades = pd.DataFrame(columns=["candidate_time","layer","entry_price","size","ret","pnl","filled_at"])

# ---------------------------
# Overlay entries on price
# ---------------------------
if overlay_entries_on_price and not trades.empty:
    fig, ax = plt.subplots(figsize=(12, 4))
    if 'close' in bars.columns:
        bars['close'].plot(ax=ax, label='close')
    trades_plot = trades.copy()
    trades_plot['candidate_time'] = pd.to_datetime(trades_plot['candidate_time'])
    trades_plot = trades_plot[trades_plot['candidate_time'].isin(bars.index)]
    for _, r in trades_plot.iterrows():
        t = r['candidate_time']
        entry_price = r.get('entry_price', None)
        color = 'g' if r.get('ret', 0) > 0 else 'r'
        ax.axvline(x=t, color=color, alpha=0.6, linewidth=0.8)
        if entry_price is not None:
            ax.plot(t, entry_price, marker='o', color=color)
    ax.set_title(f"{symbol} — Price with entry overlays (green win / red loss)")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# Compute trade metrics
# ---------------------------
trades['ret'] = pd.to_numeric(trades.get('ret', 0.0), errors='coerce').fillna(0.0)
trades['size'] = pd.to_numeric(trades.get('size', 0.0), errors='coerce').fillna(0.0)
trades['pnl'] = trades['size'] * trades['ret']

num_trades = len(trades)
total_pnl = trades['pnl'].sum() if num_trades > 0 else 0.0
avg_ret = trades['ret'].mean() if num_trades > 0 else 0.0
median_ret = trades['ret'].median() if num_trades > 0 else 0.0
std_ret = trades['ret'].std(ddof=0) if num_trades > 1 else 0.0
win_rate = (trades['ret'] > 0).sum() / num_trades if num_trades > 0 else 0.0

st.metric("Num trades (simulated)", f"{num_trades}")
st.metric("Total PnL (simulated)", f"{total_pnl:.6f}")
st.metric("Average return / filled trade", f"{avg_ret:.6f}")
st.metric("Win rate", f"{win_rate:.2%}")

if not trades.empty:
    st.dataframe(trades.head())
    pnl_series = trades.groupby('candidate_time')['pnl'].sum().cumsum()
    fig2, ax2 = plt.subplots()
    pnl_series.plot(ax=ax2)
    ax2.set_title("Cumulative PnL (simulated)")
    st.pyplot(fig2)

st.success("Demo complete.")

# ---------------------------
# Save final model
# ---------------------------
st.subheader("Save Model (train on full candidate universe)")
model_name_input = st.text_input("Enter model name", value=f"confirm_model_{symbol.replace('=','_')}")
if st.button("Save model as .model + metadata"):
    try:
        full_candidates = generate_candidates_and_labels(
            df=compute_rvol(bars, window=asset_obj.rvol_lookback),
            lookback=64,
            k_tp=2.0,
            k_sl=1.0,
            atr_window=asset_obj.atr_lookback,
            max_bars=max_bars
        )
        if full_candidates is None or full_candidates.empty:
            st.error("Full candidate generation returned empty — cannot train final model.")
        else:
            if include_health_as_feature:
                full_candidates['candidate_date'] = pd.to_datetime(full_candidates['candidate_time']).dt.normalize()
                hg = health_df[['health_gauge']].copy()
                hg = hg.reindex(pd.to_datetime(hg.index).normalize()).reset_index().rename(columns={'index':'candidate_date'})
                full_candidates = full_candidates.merge(hg, on='candidate_date', how='left')
                full_candidates['health_gauge'] = full_candidates['health_gauge'].fillna(method='ffill').fillna(0.0)
                full_candidates.drop(columns=['candidate_date'], inplace=True)

            for col in feat_cols + ["label"]:
                if col not in full_candidates.columns:
                    full_candidates[col] = np.nan
            for col in feat_cols:
                full_candidates[col] = pd.to_numeric(full_candidates[col], errors="coerce").fillna(0)

            full_clean = full_candidates.dropna(subset=["label"])
            full_clean = full_clean[full_clean["label"].isin([0, 1])]
            if full_clean.empty:
                st.error("No valid labeled data in full candidate set.")
            else:
                final_model, final_featlist, final_metrics = train_xgb_confirm(
                    clean=full_clean,
                    feature_cols=feat_cols,
                    label_col="label",
                    num_boost_round=int(num_boost),
                    early_stopping_rounds=int(early_stop),
                    test_size=float(test_size),
                    random_state=42,
                    verbose=False,
                )
                saved_paths = export_model_and_metadata(final_model, final_featlist, final_metrics,
                                                        model_basename=model_name_input,
                                                        save_fi=save_feature_importance)
                st.success(f"Saved final model. Files: {saved_paths}")
    except Exception as e:
        logger.error("Saving final model failed: %s\n%s", e, traceback.format_exc())
        st.error(f"Failed to train/save final model: {e}")

# ---------------------------
# Supabase logging
# ---------------------------
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
        "model_file": None,
        "training_params": {"num_boost_round": int(num_boost), "early_stopping_rounds": int(early_stop),
                            "test_size": float(test_size)},
        "health_thresholds": {"buy_threshold": float(buy_threshold), "sell_threshold": float(sell_threshold)},
        "p_fast": float(p_fast), "p_slow": float(p_slow), "p_deep": float(p_deep),
    }
    backtest_metrics = {
        "num_trades": int(num_trades),
        "total_pnl": float(total_pnl),
        "avg_ret": float(avg_ret),
        "median_ret": float(median_ret),
        "std_ret": float(std_ret),
        "win_rate": float(win_rate),
        "latest_health": float(latest_health),
    }
    combined_metrics = {}
    combined_metrics.update(metrics if isinstance(metrics, dict) else {})
    combined_metrics.update(backtest_metrics)

    trade_list = []
    if not trades.empty:
        for r in trades.to_dict(orient="records"):
            trade_list.append({
                "candidate_time": str(r.get("candidate_time")),
                "layer": r.get("layer", None),
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
        logger.error("Supabase log failed: %s", e)
        st.error(f"Failed to log to Supabase: {e}")

# ---------------------------
# Breadth / Grid Sweep handlers
# ---------------------------
if run_breadth:
    st.info("Running breadth backtest modes…")
    try:
        clean_for_breadth = clean if 'clean' in locals() and clean is not None and not clean.empty else None
        if clean_for_breadth is None:
            st.error("No candidate set available for breadth backtest.")
            st.stop()

        breadth_results = run_breadth_backtest(
            clean=clean_for_breadth,
            bars=bars,
            asset_obj=asset_obj,
            rr_vals=rr_vals,
            sl_ranges=sl_ranges,
            session_modes=session_modes,
            mpt_list=mpt_list,
            max_bars=int(max_bars),
            include_health=include_health_as_feature,
            health_df=health_df if 'health_df' in locals() else None,
            model_train_kwargs={
                "num_boost_round": int(num_boost),
                "early_stopping_rounds": int(early_stop),
                "test_size": float(test_size)
            }
        )

        summary_df = pd.DataFrame(breadth_results.get("summary", []))
        if not summary_df.empty:
            st.subheader("Breadth Modes Summary")
            st.dataframe(summary_df)
        else:
            st.warning("Breadth backtest returned no summary rows.")
    except Exception as exc:
        logger.error("Breadth backtest failed: %s\n%s", exc, traceback.format_exc())
        st.error(f"Breadth backtest failed: {exc}")
        st.code(traceback.format_exc())

if run_sweep_btn:
    st.info("Running grid sweep simulations…")
    try:
        clean_for_sweep = clean if 'clean' in locals() and clean is not None and not clean.empty else None
        if clean_for_sweep is None:
            st.error("No candidate set available for grid sweep.")
            st.stop()

        sweep_results = summarize_sweep(
            clean=clean_for_sweep,
            rr_vals=rr_vals,
            sl_ranges=sl_ranges,
            mpt_list=mpt_list,
            model_train_kwargs={
                "num_boost_round": int(num_boost),
                "early_stopping_rounds": int(early_stop),
                "test_size": float(test_size)
            }
        )
        st.subheader("Sweep Results")
        st.dataframe(pd.DataFrame(sweep_results))
    except Exception as exc:
        logger.error("Grid sweep failed: %s\n%s", exc, traceback.format_exc())
        st.error(f"Grid sweep failed: {exc}")
        st.code(traceback.format_exc())