# app.py — Entry-Range Triangulation Demo (chunk 1/4)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import traceback
import uuid
import io
import os
import torch
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report

# Project modules (stay consistent with your repo names)
from fetch_data import fetch_price, fetch_cot, init_socrata_client
from features import compute_rvol, calculate_health_gauge, ensure_no_duplicate_index
from labeling import generate_candidates_and_labels
from model import train_xgb_confirm, predict_confirm_prob, BoosterWrapper
from backtest import simulate_limits
# breadth and sweep modules may exist or not; import defensively
try:
    from breadth_backtest import run_breadth_backtest
except Exception:
    run_breadth_backtest = None
try:
    from sweep import run_sweep
except Exception:
    run_sweep = None

from supabase_logger import SupabaseLogger

# ---------------------------
# Page + logger setup
# ---------------------------
st.set_page_config(page_title="Entry-Range Triangulation Demo (Multi-model)", layout="wide")
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# ---------------------------
# UI inputs (controls)
# ---------------------------
st.title("Entry-Range Triangulation Demo (Multi-model / Bundled Top-3)")
with st.sidebar:
    st.header("Data / Run Controls")
    symbol = st.text_input("Symbol", value="GC=F")
    start_date = st.date_input("Start date", value=datetime.utcnow().date() - timedelta(days=90))
    end_date = st.date_input("End date", value=datetime.utcnow().date())
    interval = st.selectbox("Bar interval", options=["1m", "5m", "15m", "1h", "1d"], index=4)

    st.markdown("---")
    st.header("HealthGauge thresholds")
    buy_threshold = st.number_input("Buy threshold (HealthGauge ≥)", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
    sell_threshold = st.number_input("Sell threshold (HealthGauge ≤)", min_value=0.0, max_value=1.0, value=0.40, step=0.01)

    st.markdown("---")
    st.header("XGBoost training")
    num_boost = st.number_input("Boosting rounds", value=200, step=50)
    early_stop = st.number_input("Early stopping rounds", value=20, step=5)
    test_size = st.number_input("Test set fraction", min_value=0.01, max_value=0.5, value=0.2, step=0.01)

    st.markdown("---")
    st.header("Layer thresholds")
    p_fast = st.number_input("Fast layer prob threshold", value=0.70, min_value=0.0, max_value=1.0, step=0.01)
    p_slow = st.number_input("Slow layer prob threshold", value=0.55, min_value=0.0, max_value=1.0, step=0.01)
    p_deep = st.number_input("Deep layer prob threshold", value=0.45, min_value=0.0, max_value=1.0, step=0.01)

    st.markdown("---")
    st.header("Extras")
    include_health_as_feature = st.checkbox("Include HealthGauge as training feature", value=True)
    show_confusion = st.checkbox("Show confusion matrix / classification report", value=True)
    overlay_entries_on_price = st.checkbox("Overlay simulated entries on price chart", value=True)
    save_feature_importance = st.checkbox("Save feature importance JSON with model", value=True)
    force_run = st.checkbox("Force run pipeline (ignore HealthGauge gating)", value=False)

    st.markdown("---")
    st.header("Breadth / Sweep")
    run_breadth = st.button("Run breadth modes (Low / Mid / High)")
    run_sweep_btn = st.button("Run grid sweep")

    st.markdown("---")
    st.header("Top-3 bundling")
    bundle_after_run = st.checkbox("Automatically bundle top-3 models after run", value=True)
    topk = st.number_input("Number of top models to bundle", min_value=1, max_value=10, value=3, step=1)

# buttons
run_main = st.button("Run main pipeline (train & simulate)")

# app.py (chunk 2/4) — helpers and candidate builder

# Simple parsing helpers (if needed later)
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

def timestamp_str() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# Model + metadata exporter (saves both single-model and bundle)
def save_model_bundle(models_info: list, bundle_basename: str) -> str:
    """
    Save a list of model descriptors (dicts with wrapper, feature_list, metrics, name)
    into a single .pt bundle (torch.save). Returns file path.
    """
    ts = timestamp_str()
    fname = f"{bundle_basename}_{ts}_bundle.pt"
    payload = {"models": []}
    for mi in models_info:
        # each mi: {"name":..., "wrapper": BoosterWrapper, "feature_list": [...], "metrics": {...}, "feature_importance": {...}}
        # We won't attempt to deep-serialize xgboost objects using pickle problems; we will store:
        # - joblib serialized bytes of the wrapper (safe)
        import joblib, tempfile
        tmpf = tempfile.NamedTemporaryFile(delete=False)
        tmpf.close()
        try:
            mi_copy = dict(mi)
            # Replace wrapper with joblib dump path content bytes
            joblib.dump({"booster": mi_copy["wrapper"].booster, "feature_names": mi_copy["feature_list"]}, tmpf.name)
            with open(tmpf.name, "rb") as fh:
                serialized_bytes = fh.read()
            mi_copy["wrapper_serialized"] = serialized_bytes
            # remove the live wrapper to avoid issues
            mi_copy.pop("wrapper", None)
            payload["models"].append(mi_copy)
        finally:
            try:
                os.unlink(tmpf.name)
            except Exception:
                pass

    # Save the payload with torch (can handle bytes/content)
    torch.save(payload, fname)
    return fname

# Candidate builder (thin wrapper around labeling.generate_candidates_and_labels + features)
def build_or_fetch_candidates(bars: pd.DataFrame, lookback: int = 64, max_bars: int = 60, include_health: bool = False, health_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute rvol on bars, call labeling.generate_candidates_and_labels,
    and enrich with micro-features expected by model (tick_rate, uptick_ratio, buy_vol_ratio, micro_range, rvol_micro).
    This function keeps things defensive.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()

    # compute rvol on bars (features.compute_rvol returns pd.Series)
    try:
        bars = bars.copy()
        bars["rvol"] = compute_rvol(bars, lookback=20)
    except Exception as exc:
        logger.warning("compute_rvol failed: %s", exc)
        bars["rvol"] = 1.0

    # call labeling to produce candidates
    try:
        candidates = generate_candidates_and_labels(df=bars, lookback=lookback, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=max_bars)
    except Exception as exc:
        logger.error("generate_candidates_and_labels failed: %s", exc)
        return pd.DataFrame()

    if candidates is None or candidates.empty:
        return pd.DataFrame()

    # Basic micro-features (best-effort placeholders if missing)
    # NOTE: these can be replaced with real domain features you already compute
    candidates = candidates.reset_index(drop=True)
    if "tick_rate" not in candidates.columns:
        candidates["tick_rate"] = 1.0
    if "uptick_ratio" not in candidates.columns:
        candidates["uptick_ratio"] = 0.5
    if "buy_vol_ratio" not in candidates.columns:
        candidates["buy_vol_ratio"] = 0.5
    if "micro_range" not in candidates.columns:
        candidates["micro_range"] = (candidates["tp_price"] - candidates["sl_price"]).abs() / candidates["entry_price"]
    if "rvol_micro" not in candidates.columns:
        # align rvol to candidate timestamps
        try:
            rvol_ser = bars["rvol"].reindex(pd.to_datetime(candidates["candidate_time"])).fillna(method="ffill").fillna(1.0).values
            candidates["rvol_micro"] = rvol_ser
        except Exception:
            candidates["rvol_micro"] = 1.0

    # optionally merge health gauge
    if include_health and (health_df is not None) and (not health_df.empty):
        try:
            candidates["candidate_date"] = pd.to_datetime(candidates["candidate_time"]).dt.normalize()
            hg = health_df[["health_gauge"]].copy()
            hg = hg.reindex(pd.to_datetime(hg.index).normalize()).reset_index().rename(columns={"index": "candidate_date"})
            candidates = candidates.merge(hg, on="candidate_date", how="left")
            candidates["health_gauge"] = candidates["health_gauge"].fillna(method="ffill").fillna(0.0)
            candidates = candidates.drop(columns=["candidate_date"])
        except Exception as exc:
            logger.warning("Health merge failed: %s", exc)
            candidates["health_gauge"] = 0.0

    # ensure label exists (label is 0/1 produced by labeling)
    if "label" not in candidates.columns:
        candidates["label"] = np.nan

    return candidates

# app.py (chunk 3/4) — main run, multi-model training, simulation

# Feature columns used for confirm model
base_feat_cols = ['tick_rate', 'uptick_ratio', 'buy_vol_ratio', 'micro_range', 'rvol_micro']
if include_health_as_feature:
    base_feat_cols = base_feat_cols + ['health_gauge']

def safe_train_wrapper(clean_df: pd.DataFrame, feat_cols: list, name: str, train_kwargs: dict):
    """
    Train model and return dict with wrapper, feature_list, metrics
    """
    try:
        wrapper, featlist, metrics = train_xgb_confirm(
            clean=clean_df,
            feature_cols=feat_cols,
            label_col="label",
            num_boost_round=int(train_kwargs.get("num_boost_round", num_boost)),
            early_stopping_rounds=int(train_kwargs.get("early_stopping_rounds", early_stop)),
            test_size=float(train_kwargs.get("test_size", test_size)),
            random_state=int(train_kwargs.get("random_state", 42)),
            verbose=False,
        )
        return {"name": name, "wrapper": wrapper, "feature_list": featlist, "metrics": metrics}
    except Exception as exc:
        logger.error("Training wrapper '%s' failed: %s", name, exc)
        return {"name": name, "error": str(exc)}

# Run main pipeline
if run_main:
    logger.info("User triggered main pipeline run")
    st.info("Fetching price data…")
    try:
        bars = fetch_price(symbol, start_date.isoformat(), end_date.isoformat(), interval)
    except Exception as exc:
        logger.error("fetch_price failed: %s", exc)
        st.error(f"fetch_price failed: {exc}")
        st.stop()

    if bars is None or bars.empty:
        st.error("No price bars available for specified range.")
        st.stop()

    # daily bars + health gauge
    try:
        daily_bars = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        daily_bars = ensure_no_duplicate_index(daily_bars)
    except Exception as exc:
        logger.error("daily bars aggregation failed: %s", exc)
        daily_bars = pd.DataFrame()

    try:
        cot = pd.DataFrame()
        try:
            client = init_socrata_client()
            cot = fetch_cot(client, start=(start_date - timedelta(days=365)).isoformat(), end=end_date.isoformat())
        except Exception as e:
            logger.warning("COT fetch skipped/failed: %s", e)
        # health expects a DataFrame with rvol or similar — compute rvol on daily_bars and pass to calculate_health_gauge
        if not daily_bars.empty:
            daily_bars = daily_bars.copy()
            daily_bars["rvol"] = compute_rvol(daily_bars, lookback=asset_obj.rvol_lookback if 'asset_obj' in globals() else 20)
            health_df = calculate_health_gauge(daily_bars, rvol_col="rvol", threshold=1.5)
    except Exception as exc:
        logger.error("Health/ COT preparation failed: %s", exc)
        health_df = pd.DataFrame()

    if health_df is None or (hasattr(health_df, "empty") and health_df.empty):
        st.warning("HealthGauge empty — pipeline may halt unless forced.")
        if not force_run:
            st.stop()

    latest_health = None
    try:
        latest_health = float(health_df.iloc[-1]) if isinstance(health_df, pd.Series) else float(health_df['health_gauge'].iloc[-1]) if 'health_gauge' in health_df.columns else 0.0
    except Exception:
        latest_health = 0.0
    st.metric("Latest HealthGauge", f"{latest_health:.4f}")
    buy_allowed = latest_health >= buy_threshold
    sell_allowed = latest_health <= sell_threshold
    st.write(f"Buy allowed: {buy_allowed}, Sell allowed: {sell_allowed}")
    if not (buy_allowed or sell_allowed or force_run):
        st.warning("Health gauge not in allowed band. Halting.")
        st.stop()

    # Build candidates (clean)
    try:
        clean = build_or_fetch_candidates(bars, lookback=64, max_bars=100, include_health=include_health_as_feature, health_df=health_df)
    except Exception as exc:
        logger.error("build_or_fetch_candidates failed: %s", exc)
        st.error(f"Candidate generation failed: {exc}")
        st.stop()

    if clean is None or clean.empty:
        st.error("No labeled candidates available after generation.")
        st.stop()

    st.write("Candidates generated:", len(clean))
    st.dataframe(clean.head())

    # Primary confirm model (single model)
    st.info("Training primary confirm model")
    try:
        model_main_info = safe_train_wrapper(clean, base_feat_cols, "confirm_primary", {"num_boost_round": num_boost, "early_stopping_rounds": early_stop, "test_size": test_size})
        if "error" in model_main_info:
            raise RuntimeError(model_main_info["error"])
        # model_main_info contains wrapper, feature_list, metrics
    except Exception as exc:
        logger.error("Primary training failed: %s", exc)
        st.error(f"Primary training failed: {exc}")
        st.stop()

    st.write("Primary training metrics:")
    st.json(model_main_info.get("metrics", {}))

    # Predict probabilities using main model
    try:
        probs_main = predict_confirm_prob(model_main_info["wrapper"], clean, model_main_info["feature_list"])
        clean = clean.copy()
        clean["pred_prob_main"] = probs_main.reindex(clean.index).fillna(0.0)
        clean["pred_label_main"] = (clean["pred_prob_main"] >= p_fast).astype(int)
    except Exception as exc:
        logger.error("Primary predict failed: %s", exc)
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # Optional: confusion matrix
    if show_confusion:
        try:
            cm = confusion_matrix(clean["label"], (clean["pred_prob_main"] >= 0.5).astype(int))
            cr = classification_report(clean["label"], (clean["pred_prob_main"] >= 0.5).astype(int), output_dict=True, zero_division=0)
            st.subheader("Primary model - confusion matrix")
            st.write(pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]))
            st.subheader("Primary model - classification report")
            st.write(pd.DataFrame(cr).transpose())
        except Exception as exc:
            logger.warning("Confusion/classification generation failed: %s", exc)

    # Simulate with primary model
    st.info("Simulating fills for primary model")
    try:
        trades_primary = simulate_limits(clean, bars, label_col="pred_label_main", symbol=symbol, rr=2.0, sl=0.01, tp=0.02, max_holding=60)
    except Exception as exc:
        logger.error("simulate_limits (primary) failed: %s", exc)
        trades_primary = pd.DataFrame()

    # Compute summary metrics for primary
    def summarize_trade_df(trades_df: pd.DataFrame):
        if trades_df is None or trades_df.empty:
            return {"total_pnl": 0.0, "num_trades": 0, "win_rate": 0.0}
        trades_df = trades_df.copy()
        trades_df["pnl"] = pd.to_numeric(trades_df.get("pnl", 0.0), errors="coerce").fillna(0.0)
        return {"total_pnl": float(trades_df["pnl"].sum()), "num_trades": int(len(trades_df)), "win_rate": float((trades_df["pnl"] > 0).mean())}

    primary_summary = summarize_trade_df(trades_primary)
    st.write("Primary simulation summary:", primary_summary)
    logger.info("Primary summary: %s", primary_summary)

    # ---------------------------
    # Multi-layer training: Low / Mid / High models (simple sampling differences)
    # ---------------------------
    st.info("Training layer models (Low / Mid / High)")
    layer_train_kwargs = {"num_boost_round": num_boost, "early_stopping_rounds": early_stop, "test_size": test_size, "random_state": 42}
    layers = [
        {"name": "low_layer", "oversample_pos_frac": 1.5, "threshold": p_deep},   # conservative
        {"name": "mid_layer", "oversample_pos_frac": 1.0, "threshold": p_slow},   # balanced
        {"name": "high_layer", "oversample_pos_frac": 0.7, "threshold": p_fast},   # aggressive
    ]

    trained_layers = []
    for layer in layers:
        lname = layer["name"]
        try:
            # create training copy and apply simple sampling: oversample positives by factor
            working = clean.copy().reset_index(drop=True)
            pos = working[working["label"] == 1]
            neg = working[working["label"] == 0]
            if len(pos) == 0 or len(neg) == 0:
                logger.warning("Layer %s: insufficient class counts; skipping specialized sampling", lname)
                train_df = working
            else:
                # oversample positives
                try:
                    pos_upsampled = pos.sample(frac=max(1.0, layer["oversample_pos_frac"]), replace=True, random_state=42)
                    train_df = pd.concat([pos_upsampled, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
                except Exception:
                    train_df = working

            trained = safe_train_wrapper(train_df, base_feat_cols, lname, layer_train_kwargs)
            if "error" in trained:
                logger.warning("Layer %s training returned error: %s", lname, trained.get("error"))
                trained_layers.append({"name": lname, "error": trained.get("error")})
                continue

            # Predict probabilities for simulation using this layer's wrapper
            probs_layer = predict_confirm_prob(trained["wrapper"], clean, trained["feature_list"])
            clean[f"pred_prob_{lname}"] = probs_layer.reindex(clean.index).fillna(0.0)
            clean[f"pred_label_{lname}"] = (clean[f"pred_prob_{lname}"] >= layer["threshold"]).astype(int)

            # simulate
            trades_layer = simulate_limits(clean, bars, label_col=f"pred_label_{lname}", symbol=symbol, rr=2.0, sl=0.01, tp=0.02, max_holding=60)
            summary_layer = summarize_trade_df(trades_layer)
            trained["trades_df"] = trades_layer
            trained["summary"] = summary_layer
            trained["layer_threshold"] = layer["threshold"]
            trained_layers.append(trained)
            st.write(f"Layer {lname} summary:", summary_layer)
        except Exception as exc:
            logger.error("Layer %s failed: %s", lname, traceback.format_exc())
            trained_layers.append({"name": lname, "error": str(exc)})

    # Present layer results table
    try:
        rows = []
        for t in trained_layers:
            if "summary" in t:
                rows.append({"layer": t["name"], "total_pnl": t["summary"]["total_pnl"], "num_trades": t["summary"]["num_trades"], "win_rate": t["summary"]["win_rate"]})
        if rows:
            st.subheader("Layer summaries")
            st.dataframe(pd.DataFrame(rows))
    except Exception:
        pass

# app.py (chunk 4/4) — top-3 selection, bundling, breadth & sweep handlers, Supabase logging

# Determine top-K models by primary metric (total_pnl) then win_rate
def select_topk_models(trained_list: list, k: int = 3):
    scored = []
    for t in trained_list:
        if "summary" not in t:
            continue
        scored.append((t["name"], t["summary"]["total_pnl"], t["summary"]["win_rate"], t))
    if not scored:
        return []
    df = pd.DataFrame([{"name": n, "total_pnl": p, "win_rate": w, "obj": o} for (n,p,w,o) in scored])
    # primary sort by total_pnl desc, secondary by win_rate desc
    df_sorted = df.sort_values(["total_pnl", "win_rate"], ascending=[False, False]).reset_index(drop=True)
    topk = df_sorted.head(k).to_dict(orient="records")
    return [r["obj"] for r in topk]

# Auto bundle top-k models if requested
if bundle_after_run:
    st.info("Selecting top models to bundle...")
    top_models = select_topk_models(trained_layers, int(topk))
    if not top_models:
        st.warning("No trained layer models available to bundle.")
    else:
        # prepare model descriptors for saving
        models_info = []
        for m in top_models:
            try:
                # Get feature importance if available
                fi = {}
                try:
                    fi_df = m["wrapper"].get_feature_importance()
                    fi = dict(zip(fi_df["feature"].tolist(), fi_df["importance"].tolist()))
                except Exception:
                    fi = {}
                metrics = m.get("metrics", {})
                metrics.update(m.get("summary", {}))
                models_info.append({
                    "name": m["name"],
                    "wrapper": m["wrapper"],
                    "feature_list": m["feature_list"],
                    "metrics": metrics,
                    "feature_importance": fi
                })
            except Exception as exc:
                logger.warning("Preparing model for bundle failed: %s", exc)
        if models_info:
            bundle_basename = f"{symbol.replace('=','_')}_top{len(models_info)}"
            try:
                bundle_path = save_model_bundle(models_info, bundle_basename)
                st.success(f"Saved model bundle to {bundle_path}")
                # Offer download
                with open(bundle_path, "rb") as fh:
                    bts = fh.read()
                st.download_button("Download top models bundle (.pt)", bts, file_name=os.path.basename(bundle_path), mime="application/octet-stream")
            except Exception as exc:
                logger.error("Saving bundle failed: %s", exc)
                st.error(f"Failed to save model bundle: {exc}")

# ---------------------------
# Save final model (train on full candidate universe)
# ---------------------------
st.subheader("Save Model (train on full candidate universe)")
model_name_input = st.text_input("Enter model name", value=f"confirm_model_{symbol.replace('=','_')}")
if st.button("Save model as .model + metadata"):
    try:
        full_candidates = generate_candidates_and_labels(
            df=compute_rvol(bars, lookback=20).to_frame().rename(columns={0: "rvol"}) if False else build_or_fetch_candidates(bars, lookback=64, max_bars=100, include_health=include_health_as_feature, health_df=health_df)
        )
        # NOTE: The above line tries to reuse build_or_fetch_candidates; if your generate_candidates_and_labels signature differs,
        # replace as appropriate. We attempt to only train if valid labeled data exists.
        if full_candidates is None or full_candidates.empty:
            st.error("Full candidate generation returned empty — cannot train final model.")
        else:
            for col in base_feat_cols + ["label"]:
                if col not in full_candidates.columns:
                    full_candidates[col] = np.nan
            for col in base_feat_cols:
                full_candidates[col] = pd.to_numeric(full_candidates[col], errors="coerce").fillna(0)
            full_clean = full_candidates.dropna(subset=["label"])
            full_clean = full_clean[full_clean["label"].isin([0,1])]
            if full_clean.empty:
                st.error("No valid labeled data in full candidate set.")
            else:
                final_model, final_featlist, final_metrics = train_xgb_confirm(
                    clean=full_clean,
                    feature_cols=base_feat_cols,
                    label_col="label",
                    num_boost_round=int(num_boost),
                    early_stopping_rounds=int(early_stop),
                    test_size=float(test_size),
                    random_state=42,
                    verbose=False,
                )
                # export using joblib or xgboost's booster
                ts = timestamp_str()
                fname = f"{model_name_input}_{ts}.model"
                try:
                    final_model.booster.save_model(fname)
                except Exception:
                    # fallback: joblib
                    import joblib
                    joblib.dump(final_model, fname + ".joblib")
                    fname = fname + ".joblib"
                st.success(f"Saved final model to {fname}")
    except Exception as exc:
        logger.error("Saving final model failed: %s", exc)
        st.error(f"Saving final model failed: {exc}")

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
        "feature_cols": base_feat_cols,
        "model_file": None,
        "training_params": {"num_boost_round": int(num_boost), "early_stopping_rounds": int(early_stop), "test_size": float(test_size)},
        "health_thresholds": {"buy_threshold": float(buy_threshold), "sell_threshold": float(sell_threshold)},
        "p_fast": float(p_fast), "p_slow": float(p_slow), "p_deep": float(p_deep),
    }
    # combine metrics: primary + layers
    combined_metrics = {}
    combined_metrics.update(model_main_info.get("metrics", {}) if "model_main_info" in locals() else {})
    combined_metrics["primary_summary"] = primary_summary if 'primary_summary' in locals() else {}
    combined_metrics["layers"] = {t["name"]: t.get("summary", {}) for t in trained_layers if "name" in t}
    trade_list = []
    if 'trades_primary' in locals() and trades_primary is not None and not trades_primary.empty:
        for r in trades_primary.to_dict(orient="records"):
            trade_list.append({
                "entry_time": str(r.get("entry_time")),
                "exit_time": str(r.get("exit_time")),
                "pnl": float(r.get("pnl") or 0.0),
                "symbol": r.get("symbol"),
            })
    try:
        supa = SupabaseLogger()
        run_id_returned = supa.log_run(metrics=combined_metrics, metadata=metadata, trades=trade_list)
        st.success(f"Logged run to Supabase with run_id: {run_id_returned}")
    except Exception as exc:
        logger.error("Supabase log failed: %s", exc)
        st.error(f"Failed to log to Supabase: {exc}")

# ---------------------------
# Breadth & Sweep handlers (if user pressed their buttons)
# ---------------------------
if run_breadth:
    st.info("Running breadth backtest modes…")
    if run_breadth_backtest is None:
        st.error("run_breadth_backtest is not available in this environment.")
    else:
        try:
            clean_for_breadth = clean if (clean is not None and not clean.empty) else None
            if clean_for_breadth is None:
                st.error("No candidates for breadth run; run main pipeline first.")
            else:
                breadth_results = run_breadth_backtest(clean=clean_for_breadth, bars=bars, symbol=symbol)
                # breadth_results expected dict of mode -> {"trades": n, "overlay": df}
                # present summary
                rows = []
                for mode, res in breadth_results.items():
                    if isinstance(res, dict) and "overlay" in res and res["overlay"] is not None:
                        summary = summarize_trade_df(res["overlay"])
                        rows.append({"mode": mode, **summary})
                if rows:
                    st.subheader("Breadth summary")
                    st.dataframe(pd.DataFrame(rows))
        except Exception as exc:
            logger.error("Breadth backtest handler failed: %s", exc)
            st.error(f"Breadth backtest failed: {exc}")

if run_sweep_btn:
    st.info("Running sweep backtest…")
    if run_sweep is None:
        st.error("run_sweep is not available in this environment.")
    else:
        try:
            clean_for_sweep = clean if (clean is not None and not clean.empty) else None
            if clean_for_sweep is None:
                st.error("No candidates for sweep run; run main pipeline first.")
            else:
                sweep_results = run_sweep(clean_for_sweep, bars, symbol=symbol)
                st.subheader("Sweep results (raw)")
                st.write(sweep_results)
        except Exception as exc:
            logger.error("Sweep handler failed: %s", exc)
            st.error(f"Sweep failed: {exc}")

st.info("App ready.")