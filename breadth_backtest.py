# breadth_backtest.py
"""
Breadth/backtest helper for running Low/Mid/High breadth modes and debugging logs.

Provides:
 - run_breadth_backtest(...) : orchestrates candidate creation (if needed), model training,
   prediction, simulation and aggregation for breadth modes.
 - returns a pd.DataFrame summary and a dict of detailed trades per mode.

This module is defensive: it logs every major step and returns helpful diagnostics
so the caller (e.g. app.py) can show what went wrong.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project imports (these should exist in the repo)
try:
    from labeling import generate_candidates_and_labels
    from features import compute_rvol, calculate_health_gauge
    from model import train_xgb_confirm, predict_confirm_prob
    from backtest import simulate_limits
except Exception as e:
    # keep imports lazy/fail-safe; raise later when actually used
    generate_candidates_and_labels = None  # type: ignore
    compute_rvol = None  # type: ignore
    calculate_health_gauge = None  # type: ignore
    train_xgb_confirm = None  # type: ignore
    predict_confirm_prob = None  # type: ignore
    simulate_limits = None  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _ensure_candidates_from_bars(
    bars: pd.DataFrame,
    asset_obj: Any,
    max_bars: int,
    include_health: bool = False,
    health_df: Optional[pd.DataFrame] = None,
    lookback: int = 64,
) -> pd.DataFrame:
    """
    Helper to compute rvol, create candidates and optionally merge health gauge.
    Returns cleaned candidates (numeric features + label).
    Logs every step for debugging.
    """
    if bars is None or bars.empty:
        logger.error("_ensure_candidates_from_bars: bars is empty or None.")
        return pd.DataFrame()

    if compute_rvol is None or generate_candidates_and_labels is None:
        logger.error("_ensure_candidates_from_bars: required modules not available.")
        return pd.DataFrame()

    logger.info("Computing rvol on bars (lookback=%s)…", getattr(asset_obj, "rvol_lookback", 20))
    try:
        bars_rvol = compute_rvol(bars, window=getattr(asset_obj, "rvol_lookback", 20))
    except Exception as exc:
        logger.exception("compute_rvol failed: %s", exc)
        return pd.DataFrame()

    logger.info("Generating candidates (lookback=%d, max_bars=%d)…", lookback, max_bars)
    try:
        candidates = generate_candidates_and_labels(
            df=bars_rvol,
            lookback=lookback,
            k_tp=2.0,
            k_sl=1.0,
            atr_window=getattr(asset_obj, "atr_lookback", 14),
            max_bars=max_bars,
        )
    except Exception as exc:
        logger.exception("generate_candidates_and_labels failed: %s", exc)
        return pd.DataFrame()

    if candidates is None or candidates.empty:
        logger.warning("No candidates produced by labeling.")
        return pd.DataFrame()

    # optional health merge
    if include_health and health_df is not None and not health_df.empty:
        try:
            candidates = candidates.copy()
            candidates["candidate_date"] = pd.to_datetime(candidates["candidate_time"]).dt.normalize()
            hg = health_df[["health_gauge"]].copy()
            hg = hg.reindex(pd.to_datetime(hg.index).normalize()).reset_index().rename(columns={"index": "candidate_date"})
            candidates = candidates.merge(hg, on="candidate_date", how="left")
            candidates["health_gauge"] = candidates["health_gauge"].fillna(method="ffill").fillna(0.0)
            candidates.drop(columns=["candidate_date"], inplace=True)
        except Exception as exc:
            logger.exception("Merging HealthGauge failed: %s", exc)

    # ensure micro-features exist and numeric
    feat_cols = ["tick_rate", "uptick_ratio", "buy_vol_ratio", "micro_range", "rvol_micro"]
    for c in feat_cols + ["label"]:
        if c not in candidates.columns:
            candidates[c] = np.nan
    for c in feat_cols:
        candidates[c] = pd.to_numeric(candidates[c], errors="coerce").fillna(0.0)

    clean = candidates.dropna(subset=["label"])
    clean = clean[clean["label"].isin([0, 1])].reset_index(drop=True)
    logger.info("Candidates cleaned: rows=%d", len(clean))
    return clean


def _breadth_mode_params(mode: str) -> Dict[str, Any]:
    """
    Translate a 'Low' / 'Mid' / 'High' breadth mode into simple parameter constraints.
    Returns dict with suggested rr_grid, sl_grid, thresholds (these are suggestions used to filter/evaluate).
    """
    mode = mode.lower().strip()
    if mode == "low":
        return {"buy_threshold": 0.60, "sell_threshold": 0.40, "rr_values": [1.5, 2.0], "min_occurrence": 5}
    if mode == "mid":
        return {"buy_threshold": 0.65, "sell_threshold": 0.35, "rr_values": [2.0, 3.0], "min_occurrence": 3}
    if mode == "high":
        return {"buy_threshold": 0.70, "sell_threshold": 0.30, "rr_values": [3.0, 4.0], "min_occurrence": 1}
    # default
    return {"buy_threshold": 0.60, "sell_threshold": 0.40, "rr_values": [2.0], "min_occurrence": 1}


def run_breadth_backtest(
    clean: Optional[pd.DataFrame] = None,
    bars: Optional[pd.DataFrame] = None,
    asset_obj: Optional[Any] = None,
    rr_vals: Optional[Iterable[float]] = None,
    sl_ranges: Optional[Iterable[Tuple[float, float]]] = None,
    session_modes: Optional[Iterable[str]] = None,
    mpt_list: Optional[Iterable[float]] = None,
    max_bars: int = 60,
    include_health: bool = False,
    health_df: Optional[pd.DataFrame] = None,
    model_train_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run breadth backtests for Low/Mid/High modes in sequence and return a dict with:
        - summary: pd.DataFrame with aggregated stats per mode
        - detailed_trades: dict mode -> pd.DataFrame of trades
        - diagnostics: list[str] logs and debug info

    This function logs progress and catches exceptions to avoid crashing the caller.
    """
    logs: List[str] = []
    def _log_info(msg: str, level: str = "info"):
        ts = pd.Timestamp.utcnow().isoformat()
        entry = f"{ts} {level.upper()}: {msg}"
        logs.append(entry)
        if level == "info":
            logger.info(msg)
        elif level == "warn":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        else:
            logger.debug(msg)

    result_summary_rows: List[Dict[str, Any]] = []
    detailed_trades: Dict[str, pd.DataFrame] = {}

    modes = ["Low", "Mid", "High"]
    _log_info("Starting breadth backtest run for modes: " + ", ".join(modes))

    # Ensure we have asset object
    if asset_obj is None:
        _log_info("asset_obj not provided; some defaults will be used.", "warn")

    # If clean not provided, attempt to build from bars
    if clean is None or clean.empty:
        if bars is None or bars.empty:
            _log_info("No 'clean' candidates provided and no bars available to build them. Aborting breadth run.", "error")
            return {"summary": pd.DataFrame(), "detailed_trades": {}, "diagnostics": logs}
        _log_info("No 'clean' input – building candidates from bars.")
        clean = _ensure_candidates_from_bars(bars=bars, asset_obj=asset_obj or {}, max_bars=max_bars, include_health=include_health, health_df=health_df)
        if clean is None or clean.empty:
            _log_info("Candidate build returned empty; aborting breadth run.", "error")
            return {"summary": pd.DataFrame(), "detailed_trades": {}, "diagnostics": logs}
    else:
        _log_info(f"Using provided 'clean' DataFrame: rows={len(clean)}")

    # default grid args if None
    rr_vals = list(rr_vals) if rr_vals is not None else [1.5, 2.0, 3.0]
    sl_ranges = list(sl_ranges) if sl_ranges is not None else [(0.5, 1.0), (1.0, 2.0)]
    session_modes = list(session_modes) if session_modes is not None else ["all"]
    mpt_list = list(mpt_list) if mpt_list is not None else [0.6]

    _log_info(f"Grid params: rr_vals={rr_vals}, sl_ranges={sl_ranges}, session_modes={session_modes}, mpt_list={mpt_list}")

    # iterate modes sequentially
    for mode in modes:
        try:
            _log_info(f"Running breadth mode: {mode}")
            params = _breadth_mode_params(mode)
            rr_grid = params.get("rr_values", rr_vals)
            # For each rr in rr_grid run a lightweight simulation: filter candidates by sl-range & mpt and compute stats
            mode_trades_list: List[pd.DataFrame] = []
            for rr in rr_grid:
                for sl_low, sl_high in sl_ranges:
                    for mpt in mpt_list:
                        try:
                            # for breadth mode we interpret rr & sl as filters on realized_return relative to ATR or entry-based criteria
                            # Create a copy and compute "acceptable" TP/SL returns in absolute terms using ATR if available
                            df = clean.copy()
                            if "atr" in df.columns and "entry_price" in df.columns:
                                # compute TP & SL in price terms and then returns
                                tp_return = (rr * (df["atr"] / df["entry_price"])).fillna(0.0)
                                sl_return = ((sl_low + sl_high) / 2.0) * (df["atr"] / df["entry_price"]).fillna(0.0)
                                df["tp_return"] = tp_return
                                df["sl_return"] = sl_return
                            else:
                                # fallback: use realized_return magnitude thresholds
                                df["tp_return"] = rr * 0.001  # arbitrary fallback
                                df["sl_return"] = ((sl_low + sl_high) / 2.0) * 0.001

                            # filter by model probability threshold if available on df
                            if "pred_prob" in df.columns:
                                dff = df[df["pred_prob"] >= float(mpt)].copy()
                            else:
                                dff = df.copy()

                            # simulate fills: call simulate_limits (it will decide layers via provided probs)
                            probs_input = dff["pred_prob"] if "pred_prob" in dff.columns else None
                            trades = simulate_limits(bars=bars or pd.DataFrame(), candidates=dff, probs=probs_input, p_fast=0.9, p_slow=0.7, p_deep=0.5)

                            if trades is None or trades.empty:
                                _log_info(f"Mode {mode} rr={rr} sl=({sl_low},{sl_high}) mpt={mpt} -> no trades")
                                continue

                            # attach mode metadata
                            trades = trades.copy()
                            trades["breadth_mode"] = mode
                            trades["rr"] = rr
                            trades["sl_low"] = sl_low
                            trades["sl_high"] = sl_high
                            trades["model_prob_threshold"] = mpt
                            mode_trades_list.append(trades)
                        except Exception as exc:
                            _log_info(f"Inner simulation failed for rr={rr}, sl=({sl_low},{sl_high}), mpt={mpt}: {exc}", "warn")

            # concat mode trades
            if mode_trades_list:
                mode_trades_df = pd.concat(mode_trades_list, ignore_index=True)
                detailed_trades[mode] = mode_trades_df
                # compute summary stats for the mode
                total_pnl = mode_trades_df["pnl"].sum()
                num_trades = len(mode_trades_df)
                avg_ret = float(mode_trades_df["ret"].mean())
                win_rate = float((mode_trades_df["ret"] > 0).sum() / num_trades) if num_trades > 0 else 0.0
                median_ret = float(mode_trades_df["ret"].median()) if num_trades > 0 else 0.0
                _log_info(f"Mode {mode} results: trades={num_trades}, total_pnl={total_pnl:.6f}, win_rate={win_rate:.2%}")
            else:
                mode_trades_df = pd.DataFrame()
                total_pnl = 0.0
                num_trades = 0
                avg_ret = 0.0
                win_rate = 0.0
                median_ret = 0.0
                _log_info(f"Mode {mode} produced no trades.", "warn")

            result_summary_rows.append(
                {
                    "mode": mode,
                    "num_trades": int(num_trades),
                    "total_pnl": float(total_pnl),
                    "avg_ret": float(avg_ret),
                    "median_ret": float(median_ret),
                    "win_rate": float(win_rate),
                }
            )
        except Exception as exc:
            _log_info(f"Breadth mode {mode} failed: {exc}", "error")

    summary_df = pd.DataFrame(result_summary_rows)
    return {"summary": summary_df, "detailed_trades": detailed_trades, "diagnostics": logs}