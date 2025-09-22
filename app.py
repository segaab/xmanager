# app.py — Entry-Range Triangulation Demo (full script, chunk 1/2)
# Preserves pipeline logic and fixes breadth / sweep handlers with diagnostics.

import logging
import traceback
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
from backtest import simulate_limits
# optional sweep/breadth modules — keep imports defensive
try:
    from backtest import run_backtest, summarize_sweep  # may exist in repo
except Exception:
    run_backtest = None
    summarize_sweep = None

try:
    from breadth_backtest import run_breadth_backtest
except Exception:
    run_breadth_backtest = None

from supabase_logger import SupabaseLogger
from asset_objects import assets_list

# metrics
from sklearn.metrics import confusion_matrix, classification_report

# logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("app")

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
    out: List[Tuple[float, float]] = []
    for token in [t.strip() for t in s.split(",") if t.strip()]:
        try:
            a, b = token.split("-")
            out.append((float(a), float(b)))
        except Exception:
            continue
    return out

def parse_mpts(s: str) -> List[float]:
    out: List[float] = []
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
    paths: Dict[str, str] = {}
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_file = f"{model_basename}_{ts}.model"
    meta_file = f"{model_basename}_{ts}.json"
    fi_file = f"{model_basename}_{ts}_feature_importance.json"

    try:
        # The wrapper may expose the underlying booster as `.booster`
        booster = getattr(model_wrapper, "booster", None)
        if booster is None:
            booster = getattr(model_wrapper, "model", None)
        if booster is None:
            # fallback: pickle wrapper to .pt
            torch.save({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.pt")
            paths['pt'] = f"{model_basename}_{ts}.pt"
            with open(meta_file, "w") as f:
                json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2)
            paths['meta'] = meta_file
            return paths

        # save booster in xgb native format
        booster.save_model(model_file)
        paths['model'] = model_file

        fi: Dict[str, float] = {}
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

    except Exception:
        # fallback single-file torch save
        torch.save({'model_wrapper': model_wrapper, 'features': feature_list, 'metrics': metrics}, f"{model_basename}_{ts}.pt")
        paths['pt'] = f"{model_basename}_{ts}.pt"
        with open(meta_file, "w") as f:
            json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, indent=2)
        paths['meta'] = meta_file

    return paths

# ---------------------------
# Helper to ensure we have 'clean' (labeled candidates) available
# ---------------------------
def build_or_fetch_candidates(bars: pd.DataFrame, asset_obj, max_bars_val: int, include_health: bool, health_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate candidates and return a cleaned DataFrame suitable for training/backtesting.
    """
    try:
        bars_rvol = compute_rvol(bars, window=asset_obj.rvol_lookback)
    except Exception as exc:
        logger.error("compute_rvol failed: %s", exc)
        raise

    candidates = generate_candidates_and_labels(
        df=bars_rvol,
        lookback=64,
        k_tp=2.0,
        k_sl=1.0,
        atr_window=asset_obj.atr_lookback,
        max_bars=max_bars_val
    )
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    if include_health and health_df is not None and not health_df.empty:
        candidates = candidates.copy()
        candidates['candidate_date'] = pd.to_datetime(candidates['candidate_time']).dt.normalize()
        hg = health_df[['health_gauge']].copy()
        hg = hg.reindex(pd.to_datetime(hg.index).normalize()).reset_index().rename(columns={'index':'candidate_date'})
        candidates = candidates.merge(hg, on='candidate_date', how='left')
        candidates['health_gauge'] = candidates['health_gauge'].fillna(method='ffill').fillna(0.0)
        candidates.drop(columns=['candidate_date'], inplace=True)

    # Ensure micro-features exist and are numeric
    feat_cols = ['tick_rate','uptick_ratio','buy_vol_ratio','micro_range','rvol_micro']
    if include_health:
        feat_cols = feat_cols + ['health_gauge']
    for col in feat_cols + ["label"]:
        if col not in candidates.columns:
            candidates[col] = np.nan
    for col in feat_cols:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(0)
    clean = candidates.dropna(subset=["label"])
    clean = clean[clean["label"].isin([0, 1])].reset_index(drop=True)
    return clean

# app.py — Entry-Range Triangulation Demo (full script, chunk 2a/2)
# continuation: main pipeline + training + backtest

# ---------------------------
# Main pipeline
# ---------------------------
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
        daily_bars = bars.resample("1D").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna()
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

    # dedupe index safely
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

    # Build candidates (clean)
    st.info("Computing RVol and generating candidate events…")
    try:
        clean = build_or_fetch_candidates(
            bars, asset_obj, int(max_bars), include_health_as_feature, health_df
        )
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

# app.py — Entry-Range Triangulation Demo (full script, chunk 2b/2)
# continuation from prediction section

    # ------------------------
    # Evaluation / Backtest
    # ------------------------
    st.info("Evaluating predictions with confusion matrix and metrics…")
    try:
        cm = confusion_matrix(clean['label'], clean['pred_label'])
        st.write("Confusion Matrix:", cm.tolist())
        logger.info("Confusion matrix computed.")
    except Exception as e:
        logger.error("Confusion matrix failed: %s", e)
        st.error(f"Confusion matrix computation failed: {e}")

    # Backtest with price overlays
    st.info("Running backtest with price overlays…")
    try:
        overlay = simulate_limits(clean, bars, label_col="pred_label", symbol=symbol)
        if overlay is not None and not overlay.empty:
            st.line_chart(overlay)
            st.success("Backtest overlay complete.")
        else:
            st.warning("Overlay returned empty results.")
    except Exception as e:
        logger.error("simulate_limits failed: %s", e)
        st.error(f"Backtest overlay failed: {e}")

    # Metrics summary
    try:
        prec = precision_score(clean['label'], clean['pred_label'])
        rec = recall_score(clean['label'], clean['pred_label'])
        f1 = f1_score(clean['label'], clean['pred_label'])
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall", f"{rec:.3f}")
        st.metric("F1", f"{f1:.3f}")
        logger.info("Metrics -> Precision=%.3f Recall=%.3f F1=%.3f", prec, rec, f1)
    except Exception as e:
        logger.warning("Metrics calculation failed: %s", e)

    # Save model option
    if save_model:
        try:
            import joblib
            joblib.dump(model, "xgb_confirm_model.pkl")
            st.success("Model saved to xgb_confirm_model.pkl")
            logger.info("Model persisted successfully.")
        except Exception as e:
            logger.error("Model save failed: %s", e)
            st.error(f"Model save failed: {e}")

# ----------------------------
# Breadth Backtest Handler
# ----------------------------
if breadth_mode:
    st.header("Breadth Backtest Mode")
    try:
        breadth_backtest()
        st.success("Breadth backtest completed.")
    except Exception as e:
        logger.error("Breadth backtest failed: %s", e, exc_info=True)
        st.error(f"Breadth backtest failed: {e}")

# ----------------------------
# Sweep Mode Handler
# ----------------------------
if sweep_mode:
    st.header("Sweep Mode")
    try:
        sweep()
        st.success("Sweep run completed.")
    except Exception as e:
        logger.error("Sweep run failed: %s", e, exc_info=True)
        st.error(f"Sweep run failed: {e}")

# ----------------------------
# End of Script
# ----------------------------
logger.info("Streamlit app execution finished.")