# app.py (part 1/2) — Entry-Range Triangulation Demo with enhanced reporting, portability, and HealthGauge integration
# Preserves existing pipeline logic; adds:
#  - price + entries overlay chart
#  - confusion matrix & classification report
#  - model export: xgboost model file + JSON metadata & feature importance
#  - optional inclusion of HealthGauge as a feature in the confirm model

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import uuid
import json
import torch
from datetime import date, timedelta, datetime
from typing import List, Tuple, Dict, Any

# project modules (preserve existing codebase structure)
from fetch_data import fetch_price, init_socrata_client, fetch_cot
from features import compute_rvol, calculate_health_gauge
from labeling import generate_candidates_and_labels
from model import train_xgb_confirm, predict_confirm_prob
from backtest import simulate_limits, run_backtest, summarize_sweep
from breadth_backtest import run_breadth_backtest
from supabase_logger import SupabaseLogger
from asset_objects import assets_list

# metrics
from sklearn.metrics import confusion_matrix, classification_report

# Streamlit page config
st.set_page_config(layout="wide", page_title="Entry Triangulation Demo")
st.title("Entry-Range Triangulation Demo (HealthGauge → Entry → Confirm)")

# ---------------------------
# Sidebar controls
# ---------------------------
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
    st.header("Extras & Reporting")
    include_health_as_feature = st.checkbox("Include HealthGauge as training feature", value=False)
    show_confusion = st.checkbox("Show confusion matrix / classification report", value=True)
    overlay_entries_on_price = st.checkbox("Overlay simulated entries on price chart", value=True)
    save_feature_importance = st.checkbox("Save feature importance JSON with model", value=True)

    st.markdown("---")
    st.header("Grid Sweep / Breadth")
    rr_vals = st.multiselect("RR values", options=[1.5, 2.0, 2.5, 3.0, 4.0], default=[2.0, 3.0])
    sl_input = st.text_input("SL ranges (e.g. 0.5-1.0,1.0-2.0)", value="0.5-1.0,1.0-2.0")
    session_modes = st.multiselect("Session modes", options=["all", "top_k:3", "top_k:5"], default=["all"])
    mpt_input = st.text_input("Model prob thresholds (comma-separated)", value="0.6,0.7")
    max_bars = st.number_input("Max bars horizon (labels/sim)", value=60, step=1)

    st.markdown("---")
    force_run = st.checkbox("Force run pipeline (ignore HealthGauge gating)", value=False)
    run = st.button("Run demo pipeline")
    run_breadth = st.button("Run breadth modes (Low → Mid → High)")
    run_sweep_btn = st.button("Run grid sweep (train + simulate)")

# ---------------------------
# Helper parsing functions
# ---------------------------
def parse_sl_ranges(s: str) -> List[Tuple[float, float]]:
    out = []
    for token in [t.strip() for t in s.split(",") if t.strip()]:
        try:
            a, b = token.split("-")
            out.append((float(a), float(b)))
        except Exception:
            continue
    return out

def parse_mpts(s: str) -> List[float]:
    out = []
    for token in [t.strip() for t in s.split(",") if t.strip()]:
        try:
            out.append(float(token))
        except Exception:
            continue
    return out

sl_ranges = parse_sl_ranges(sl_input)
mpt_list = parse_mpts(mpt_input)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ---------------------------
# Small helper: save model and metadata/fi
# ---------------------------
def export_model_and_metadata(model_wrapper, feature_list: List[str], metrics: Dict[str, Any], model_basename: str, save_fi: bool = True):
    """
    Save xgboost booster to .model (xgb native) + JSON metadata incl feature importance.
    Returns paths saved.
    """
    paths = {}
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_file = f"{model_basename}_{ts}.model"
    meta_file = f"{model_basename}_{ts}.json"
    fi_file = f"{model_basename}_{ts}_feature_importance.json"

    try:
        booster = getattr(model_wrapper, "booster", None)
        if booster is None:
            booster = getattr(model_wrapper, "model", None)
        if booster is None:
            torch.save({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.pt")
            paths['pt'] = f"{model_basename}_{ts}.pt"
            with open(meta_file, "w") as f:
                json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2)
            paths['meta'] = meta_file
            return paths

        booster.save_model(model_file)
        paths['model'] = model_file

        fi = {}
        try:
            fi_raw = booster.get_score(importance_type="gain")
            fi = {f: float(fi_raw.get(f, 0.0)) for f in feature_list}
        except Exception:
            fi = {f: 0.0 for f in feature_list}

        meta = {"features": feature_list, "metrics": metrics, "saved_at": ts}
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        paths['meta'] = meta_file

        if save_fi:
            with open(fi_file, "w") as f:
                json.dump(fi, f, indent=2)
            paths['feature_importance'] = fi_file

    except Exception as exc:
        torch.save({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.pt")
        paths['pt'] = f"{model_basename}_{ts}.pt"
        with open(meta_file, "w") as f:
            json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2)
        paths['meta'] = meta_file

    return paths

# ---------------------------
# Main pipeline
# ---------------------------
if run:
    st.info(f"Fetching price data for {symbol}…")
    try:
        bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        st.stop()

    if bars.empty:
        st.error("No price data returned. Check symbol and time range.")
        st.stop()
    st.success(f"Fetched {len(bars)} bars.")

    st.info("Fetching COT & computing HealthGauge…")
    try:
        client = init_socrata_client()
        cot = fetch_cot(client, start=(start_date - timedelta(days=365)).isoformat(), end=end_date.isoformat())
    except Exception as e:
        st.warning(f"COT fetch failed: {e}. Continuing with empty COT.")
        cot = pd.DataFrame()

    daily_bars = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    daily_bars = daily_bars.loc[~daily_bars.index.duplicated(keep="first")]

    health_df = calculate_health_gauge(cot, daily_bars)
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

    if not (buy_allowed or sell_allowed or force_run):
        st.warning("HealthGauge not in buy/sell band. Pipeline halted.")
        st.stop()

    st.info("Computing RVol and generating candidate events…")
    bars_rvol = compute_rvol(bars, window=asset_obj.rvol_lookback)

    try:
        candidates = generate_candidates_and_labels(
            df=bars_rvol,
            lookback=64,
            k_tp=2.0,
            k_sl=1.0,
            atr_window=asset_obj.atr_lookback,
            max_bars=max_bars
        )
    except Exception as e:
        st.error(f"Error generating candidates: {e}")
        st.stop()

    st.write("Candidates generated:", len(candidates))
    if len(candidates) == 0:
        st.warning("No candidates. Try a different date range or interval.")
        st.stop()
    st.dataframe(candidates.head())

    # Merge HealthGauge if selected
    if include_health_as_feature:
        candidates = candidates.copy()
        candidates['candidate_date'] = pd.to_datetime(candidates['candidate_time']).dt.normalize()
        hg = health_df[['health_gauge']].copy()
        hg = hg.reindex(pd.to_datetime(hg.index).normalize()).reset_index().rename(columns={'index':'candidate_date'})
        candidates = candidates.merge(hg, on='candidate_date', how='left')
        candidates['health_gauge'] = candidates['health_gauge'].fillna(method='ffill').fillna(0.0)
        candidates.drop(columns=['candidate_date'], inplace=True)

    st.info("Training XGBoost confirm model…")
    feat_cols = ['tick_rate','uptick_ratio','buy_vol_ratio','micro_range','rvol_micro']
    if include_health_as_feature:
        feat_cols.append('health_gauge')

    for col in feat_cols + ["label"]:
        if col not in candidates.columns:
            candidates[col] = np.nan
    for col in feat_cols:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(0)

    clean = candidates.dropna(subset=["label"])
    clean = clean[clean["label"].isin([0, 1])]
    if clean.empty:
        st.error("No valid labeled candidates after cleaning.")
        st.stop()

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
        st.error(f"Training failed: {e}")
        st.stop()

    st.write("Training metrics (summary):")
    st.write(metrics)

if show_confusion:
        st.subheader("Confusion Matrix & Classification Report")
        try:
            y_true = metrics.get("y_true", [])
            y_pred = metrics.get("y_pred", [])
            if len(y_true) > 0 and len(y_pred) > 0:
                cm = confusion_matrix(y_true, y_pred)
                st.write("Confusion Matrix:")
                st.write(cm)
                cr = classification_report(y_true, y_pred, output_dict=True)
                st.json(cr)
            else:
                st.warning("Metrics missing y_true/y_pred for confusion matrix.")
        except Exception as e:
            st.warning(f"Confusion matrix error: {e}")

    st.info("Simulating entries via probability thresholds…")
    try:
        sim_df = simulate_limits(
            df=clean,
            model=model,
            feature_cols=feature_list,
            p_fast=p_fast,
            p_slow=p_slow,
            p_deep=p_deep,
            rr_list=rr_vals,
            sl_ranges=sl_ranges,
            max_bars=max_bars
        )
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()

    st.write("Simulation results (head):")
    st.dataframe(sim_df.head())

    if overlay_entries_on_price:
        st.subheader("Price + Simulated Entries Overlay")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(bars.index, bars["close"], label="Close", alpha=0.8)
        buys = sim_df[sim_df["signal"] == 1]
        sells = sim_df[sim_df["signal"] == -1]
        ax.scatter(buys["time"], buys["entry"], marker="^", c="g", label="Buys", alpha=0.7)
        ax.scatter(sells["time"], sells["entry"], marker="v", c="r", label="Sells", alpha=0.7)
        ax.legend()
        ax.set_title(f"Price with Simulated Entries — {symbol}")
        st.pyplot(fig)

    # Export model + metadata
    st.info("Exporting model and metadata…")
    paths = export_model_and_metadata(model, feature_list, metrics, f"confirm_model_{symbol}", save_fi=save_feature_importance)
    st.write("Files saved:", paths)

    # Allow download
    for label, path in paths.items():
        try:
            with open(path, "rb") as f:
                st.download_button(f"Download {label}", f.read(), file_name=path)
        except Exception as e:
            st.warning(f"Download failed for {path}: {e}")

# ---------------------------
# Breadth backtest
# ---------------------------
if run_breadth:
    st.subheader("Breadth Backtest Across Session Modes")
    try:
        results = run_breadth_backtest(
            df=bars,
            candidates=candidates,
            feature_cols=feat_cols,
            rr_list=rr_vals,
            sl_ranges=sl_ranges,
            max_bars=max_bars,
            session_modes=session_modes,
            p_thresholds=mpt_list,
        )
        st.success("Breadth backtest complete.")
        st.dataframe(results.head())
        st.line_chart(results.set_index("mode")["sharpe"].fillna(0))
    except Exception as e:
        st.error(f"Breadth backtest failed: {e}")
        st.text(traceback.format_exc())

# ---------------------------
# Grid sweep
# ---------------------------
if run_sweep_btn:
    st.subheader("Parameter Sweep Results")
    try:
        sweep_df = summarize_sweep(
            df=bars,
            candidates=candidates,
            feature_cols=feat_cols,
            rr_list=rr_vals,
            sl_ranges=sl_ranges,
            mpt_list=mpt_list,
            max_bars=max_bars,
        )
        st.success("Sweep complete.")
        st.dataframe(sweep_df.head())
        st.bar_chart(sweep_df.set_index("config")["sharpe"].fillna(0))
    except Exception as e:
        st.error(f"Sweep failed: {e}")
        st.text(traceback.format_exc())

# ---------------------------
# Supabase logging
# ---------------------------
try:
    supabase_logger = SupabaseLogger()
    supabase_logger.log_run(
        run_id=str(uuid.uuid4()),
        asset=selected_asset,
        start_date=str(start_date),
        end_date=str(end_date),
        metrics=metrics,
        extra={"paths": paths if run else {}}
    )
    st.success("Run logged to Supabase.")
except Exception as e:
    st.warning(f"Supabase logging failed: {e}")