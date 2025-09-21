# app.py (part 1/2)
# Entry-Range Triangulation Demo with enhanced reporting & exports
# Preserves existing pipeline logic and adds a few robustness fixes.

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
    from backtest import run_backtest, summarize_sweep  # may exist in your repo
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

# app.py (part 2/2)
# continuation: main pipeline + reporting overlays & exports

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

    clean = candidates.dropna(subset=["label"]).reset_index(drop=True)
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

    st.info("Predicting confirm probabilities on candidate events…")
    try:
        # Use the wrapper-friendly predict_confirm_prob API: (model, candidates_df, feature_cols)
        probs = predict_confirm_prob(model, clean, feature_list)
        clean = clean.copy()
        clean['pred_prob'] = probs.reindex(clean.index).fillna(0.0)
        # predicted label uses p_fast threshold by default (can be exposed in UI)
        clean['pred_label'] = (clean['pred_prob'] >= p_fast).astype(int)
    except Exception as e:
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
            st.warning(f"Could not compute confusion/classification report: {exc}")

    st.info("Running simulated fills & backtest…")
    try:
        trades = simulate_limits(bars, clean, probs, p_fast=p_fast, p_slow=p_slow, p_deep=p_deep)
    except Exception as e:
        st.error(f"Backtest simulation failed: {e}")
        st.stop()

    st.write("Simulated trades:", 0 if trades is None else len(trades))
    if trades is None or trades.empty:
        st.warning("No trades simulated.")
        trades = pd.DataFrame(columns=["candidate_time","layer","entry_price","size","ret","pnl","filled_at"])

    # Show overlay of entries on price chart (entry markers + win/loss color)
    if overlay_entries_on_price and not trades.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        # ensure price series plotted
        if 'close' in bars.columns:
            bars['close'].plot(ax=ax, label='close')
        # only plot trades that are inside the price timeframe
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

    # compute simple trade metrics
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
    # Save final model: retrain on FULL candidate universe (ignore HealthGauge gating)
    # ---------------------------
    st.subheader("Save Model (train on full candidate universe)")
    model_name_input = st.text_input("Enter model name", value=f"confirm_model_{symbol.replace('=','_')}")
    if st.button("Save model as .model + metadata"):
        try:
            full_candidates = generate_candidates_and_labels(
                df=bars_rvol,
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
                    full_candidates = full_candidates.copy()
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
                    saved_paths = export_model_and_metadata(final_model, final_featlist, final_metrics, model_basename=model_name_input, save_fi=save_feature_importance)
                    st.success(f"Saved final model. Files: {saved_paths}")
        except Exception as e:
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
            "training_params": {"num_boost_round": int(num_boost), "early_stopping_rounds": int(early_stop), "test_size": float(test_size)},
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
            st.error(f"Failed to log to Supabase: {e}")

# ---------------------------
# Breadth / Grid Sweep modes
# ---------------------------
if run_breadth:
    st.info("Running breadth backtest modes…")
    if run_breadth_backtest is None:
        st.error("breadth_backtest.run_breadth_backtest not available in this environment.")
    else:
        try:
            breadth_results = run_breadth_backtest(clean, rr_vals=rr_vals, sl_ranges=sl_ranges, session_modes=session_modes, mpt_list=mpt_list)
            st.dataframe(breadth_results)
        except Exception as e:
            st.error(f"Breadth backtest failed: {e}")

if run_sweep_btn:
    st.info("Running grid sweep simulations…")
    if summarize_sweep is None:
        st.error("summarize_sweep not available in this environment.")
    else:
        try:
            sweep_summary = summarize_sweep(clean, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list)
            st.dataframe(sweep_summary)
        except Exception as e:
            st.error(f"Grid sweep failed: {e}")