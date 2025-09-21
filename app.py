# app.py (full file) — regenerated, preserving existing logic and adding display for sweep & fold metrics
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import uuid
import hashlib
import json
import torch
from datetime import date, timedelta, datetime

# project modules (preserve existing codebase structure)
from fetch_data import fetch_price, init_socrata_client, fetch_cot
from features import compute_rvol, calculate_health_gauge
from labeling import generate_candidates_and_labels
from model import train_xgb_confirm, predict_confirm_prob
from backtest import simulate_limits, run_backtest, summarize_sweep
from supabase_logger import SupabaseLogger
from asset_objects import assets_list

# Streamlit page config
st.set_page_config(layout="wide", page_title="Entry Triangulation Demo")
st.title("Entry-Range Triangulation Demo (HealthGauge → Entry → Confirm)")

# ---------------------------
# Sidebar controls (preserve existing)
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
    st.header("Grid Sweep (optional)")
    rr_vals = st.multiselect("RR values", options=[1.5, 2.0, 2.5, 3.0, 4.0], default=[2.0, 3.0])
    sl_input = st.text_input("SL ranges (e.g. 0.5-1.0,1.0-2.0)", value="0.5-1.0,1.0-2.0")
    session_modes = st.multiselect("Session modes", options=["all", "top_k:3", "top_k:5"], default=["all"])
    mpt_input = st.text_input("Model prob thresholds (comma-separated)", value="0.6,0.7")
    max_bars = st.number_input("Max bars horizon (labels/sim)", value=60, step=1)

    st.markdown("---")
    force_run = st.checkbox("Force run pipeline (ignore HealthGauge gating)", value=False)
    run = st.button("Run demo pipeline")
    run_sweep_btn = st.button("Run grid sweep (train + simulate)")

# ---------------------------
# Helpers: parse SL ranges & MPTs
# ---------------------------
def parse_sl_ranges(s: str):
    out = []
    for token in [t.strip() for t in s.split(",") if t.strip()]:
        try:
            a, b = token.split("-")
            out.append((float(a), float(b)))
        except Exception:
            continue
    return out

def parse_mpts(s: str):
    out = []
    for token in [t.strip() for t in s.split(",") if t.strip()]:
        try:
            out.append(float(token))
        except Exception:
            continue
    return out

sl_ranges = parse_sl_ranges(sl_input)
mpt_list = parse_mpts(mpt_input)

# ---------------------------
# Caching wrapper for sweep (tight hash)
# ---------------------------
@st.cache_data(show_spinner=False)
def cached_run_backtest(bars_hash: str, rr_vals, sl_ranges, session_modes, mpt_list,
                        max_bars_val, rvol_threshold, model_train_kwargs, feature_cols):
    """
    Cached wrapper that calls run_backtest(...) using outer-scope 'bars'.
    Keyed by bars_hash + args.
    """
    # local import to reduce top-level coupling
    from backtest import run_backtest
    return run_backtest(
        bars=bars,
        feature_cols=list(feature_cols),
        rr_grid=list(rr_vals),
        sl_grid=list(sl_ranges),
        session_modes=list(session_modes),
        model_prob_thresholds=list(mpt_list),
        max_bars=int(max_bars_val),
        rvol_threshold=float(rvol_threshold),
        train_on_allowed_session=True,
        model_train_kwargs=dict(model_train_kwargs or {}),
    )

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

    # Build daily bars and ensure no duplicate index entries
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
            bars_rvol,
            lookback=64,
            k_tp=2.0,
            k_sl=1.0,
            atr_window=asset_obj.atr_lookback,
            max_bars=max_bars
        )
    except TypeError:
        # fall back to RR/SL parameter names if labeling uses rr/sl_pct_range
        candidates = generate_candidates_and_labels(
            bars_rvol,
            lookback=64,
            rr=2.0,
            sl_pct_range=(0.5, 2.0),
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

    st.info("Training XGBoost confirm model…")
    feat_cols = ['tick_rate','uptick_ratio','buy_vol_ratio','micro_range','rvol_micro']
    # ensure features present (fill zeros where missing)
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
    # pretty display of returned metrics (folds / aggregates)
    if isinstance(metrics, dict):
        if "fold_metrics" in metrics:
            st.subheader("Per-fold metrics")
            st.dataframe(pd.DataFrame(metrics["fold_metrics"]))
        if "fold_aggregates" in metrics:
            st.subheader("Fold aggregates")
            st.write(metrics["fold_aggregates"])
        if "final_validation" in metrics:
            st.subheader("Final validation")
            st.write(metrics["final_validation"])
    else:
        st.write(metrics)

    st.info("Predicting confirm probabilities and running backtest…")
    try:
        probs = predict_confirm_prob(model, clean, feature_list)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    trades = simulate_limits(bars, clean, probs, p_fast=p_fast, p_slow=p_slow, p_deep=p_deep)
    st.write("Simulated trades:", len(trades))

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
        st.warning("No trades simulated.")
        num_trades = 0; total_pnl = 0.0; avg_ret = 0.0; win_rate = 0.0

    st.success("Demo complete.")

    # Save model as .pt (preserve behavior)
    st.subheader("Save Model")
    model_name_input = st.text_input("Enter model name", value=f"confirm_model_{symbol.replace('=','_')}")
    if st.button("Save model as .pt"):
        if 'model' in locals() and 'feature_list' in locals():
            model_fname = f"{model_name_input}.pt"
            torch.save({'model': model, 'features': feature_list}, model_fname)
            st.success(f"Saved model to {model_fname}")
        else:
            st.error("No trained model found. Run training first.")

    # Supabase logging (preserve)
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
            "num_trades": int(num_trades) if num_trades else 0,
            "total_pnl": float(total_pnl) if num_trades else 0.0,
            "avg_ret": float(avg_ret) if num_trades else 0.0,
            "median_ret": float(median_ret) if num_trades else 0.0,
            "std_ret": float(std_ret) if num_trades else 0.0,
            "win_rate": float(win_rate),
            "latest_health": float(latest_health),
        }
        combined_metrics = {}
        # merge training metrics intelligently
        if isinstance(metrics, dict):
            combined_metrics.update(metrics.get("fold_aggregates", {}))
            combined_metrics.update({"final_validation": metrics.get("final_validation", {})})
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
# Grid sweep UI and execution (preserve earlier behavior, no example calls)
# ---------------------------
if run_sweep_btn:
    # Validate grid inputs
    if not rr_vals:
        st.error("Select at least one RR value for the sweep.")
        st.stop()
    if not sl_ranges:
        st.error("Provide at least one SL range in the SL ranges input.")
        st.stop()
    if not mpt_list:
        st.error("Provide at least one model probability threshold.")
        st.stop()

    # small grid safety
    est_size = max(1, len(rr_vals)) * max(1, len(sl_ranges)) * max(1, len(session_modes)) * max(1, len(mpt_list))
    if est_size > 200:
        st.warning("Large sweep requested. Consider reducing grid size to avoid long runs.")

    # Ensure we have bars from prior run (or fetch anew)
    try:
        bars  # noqa: F821
    except NameError:
        st.info("Fetching price data first for sweep...")
        try:
            bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        except Exception as e:
            st.error(f"Error fetching price for sweep: {e}")
            st.stop()
        if bars.empty:
            st.error("No price data for sweep.")
            st.stop()

    # Create deterministic bars hash (head+tail JSON)
    head_tail = pd.concat([bars.head(5), bars.tail(5)])
    bars_hash = hashlib.sha256(head_tail.to_json().encode()).hexdigest()

    model_train_kwargs = {"num_boost_round": int(num_boost), "test_size": float(test_size)}
    feature_cols = feat_cols

    st.info("Running grid sweep (training + simulation). This may take time.")
    try:
        sweep_out = cached_run_backtest(
            bars_hash,
            rr_vals=tuple(rr_vals),
            sl_ranges=tuple(sl_ranges),
            session_modes=tuple(session_modes),
            mpt_list=tuple(mpt_list),
            max_bars_val=int(max_bars),
            rvol_threshold=asset_obj.rvol_lookback if hasattr(asset_obj, "rvol_lookback") else 1.5,
            model_train_kwargs=model_train_kwargs,
            feature_cols=tuple(feature_cols),
        )
    except Exception as e:
        st.error(f"Sweep failed: {e}")
        st.stop()

    # Present summary
    summary = sweep_out.get("summary", [])
    summary_df = pd.DataFrame(summary)
    if summary_df.empty:
        st.warning("Sweep completed but produced no summary rows.")
    else:
        st.subheader("Grid Sweep Summary")
        st.dataframe(summary_df)

        # CSV download
        csv_buf = io.StringIO()
        summary_df.to_csv(csv_buf, index=False)
        st.download_button("Download summary CSV", csv_buf.getvalue(), file_name="grid_summary.csv", mime="text/csv")

        # show simple heatmap if columns available
        if {'rr', 'model_prob_threshold', 'win_rate'}.issubset(summary_df.columns):
            pivot = summary_df.pivot_table(index='rr', columns='model_prob_threshold', values='win_rate', aggfunc='mean')
            if not pivot.empty:
                fig, ax = plt.subplots()
                im = ax.imshow(pivot.fillna(np.nan).values, aspect='auto')
                ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45)
                ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
                ax.set_title("Win rate (RR × model_prob_threshold)")
                plt.colorbar(im, ax=ax, fraction=0.03)
                st.pyplot(fig)

        # detailed trades inspection
        detailed = sweep_out.get("detailed_trades", {}) or {}
        if detailed:
            st.subheader("Inspect detailed trades")
            grid_keys = list(detailed.keys())
            sel = st.selectbox("Select grid cell", options=grid_keys)
            trades_df = detailed.get(sel)
            if trades_df is not None and not trades_df.empty:
                st.dataframe(trades_df.head(200))
                csv_tr = trades_df.to_csv(index=False)
                st.download_button("Download trades CSV", csv_tr, file_name=f"trades_{sel}.csv", mime="text/csv")
                # boxplot of returns by session
                try:
                    gp = trades_df.groupby("session")["ret"].apply(list)
                    if not gp.empty:
                        fig2, ax2 = plt.subplots(figsize=(max(6, len(gp)), 4))
                        ax2.boxplot([gp.loc[s] for s in gp.index], labels=[str(int(s)) for s in gp.index], showfliers=False)
                        ax2.set_title("Return distribution by session")
                        ax2.set_xlabel("session"); ax2.set_ylabel("return")
                        st.pyplot(fig2)
                except Exception:
                    pass
            else:
                st.info("No trades recorded for this grid cell.")
        else:
            st.info("No detailed trade data available from sweep results.")

    st.success("Grid sweep finished.")
