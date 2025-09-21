# backtest.py
"""
Backtesting & sweep utilities for Entry-Range Triangulation demo.

Provides:
 - simulate_limits(...)            : simulate fills based on confirm probabilities + realized returns
 - summarize_sweep(...)            : quick grid-style summary (RR × model_prob_threshold)
 - run_backtest(...)               : lightweight driver to run a grid and return detailed trades
 - run_breadth_backtest(...)       : run Low/Mid/High breadth modes and return aggregate stats

Notes / simplifications:
 - This module intentionally keeps interfaces simple and robust for the demo app.
 - It expects `candidates` to contain at least:
     ['candidate_time','entry_price','atr','realized_return','end_time']
   and micro-features (optional).
 - `realized_return` is used as the ground-truth filled return for a candidate when available.
 - For sweep/grid operations that change RR, we compute the implied target return:
     target_ret = (rr * atr) / entry_price
   and treat a candidate as a "win" for that RR iff realized_return >= target_ret.
 - This is not a full re-labelling engine — it's a lightweight approach that matches the
   demo app's expectations while remaining fast and deterministic.
"""

from __future__ import annotations
import logging
from typing import Optional, Union, Sequence, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------------
def _ensure_datetime_index(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars = bars.copy()
        bars.index = pd.to_datetime(bars.index)
    return bars


# -------------------------------------------------------------------------
def simulate_limits(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    probs: Union[pd.Series, dict, list, np.ndarray, None] = None,
    p_fast: float = 0.7,
    p_slow: float = 0.55,
    p_deep: float = 0.45,
) -> pd.DataFrame:
    """
    Simulate fills using candidate realized returns and confirm probabilities.

    Parameters
    ----------
    bars : pd.DataFrame
        Price bars (used primarily for timeline alignment).
    candidates : pd.DataFrame
        Rows representing candidate events. Must contain 'candidate_time' (or index),
        'entry_price' and preferably 'realized_return' and 'end_time'.
    probs : pd.Series | dict | list | np.ndarray | None
        Confirm probabilities aligned to candidates. Accepts multiple shapes:
        - pd.Series indexed like candidates
        - dict mapping index or candidate_time -> prob
        - list/np.array aligned in order with candidates.iterrows()
        - None => zeros
    p_fast, p_slow, p_deep : float
        Probability thresholds to assign layers. p_fast > p_slow > p_deep.

    Returns
    -------
    trades_df : pd.DataFrame
        DataFrame of simulated trades with columns:
          ['candidate_time','open_time','close_time','layer','entry_price',
           'size','ret','pnl','filled_at','win']
    """
    bars = _ensure_datetime_index(bars)
    trades: List[Dict[str, Any]] = []

    # Build probability map
    prob_map: Dict = {}
    if probs is None:
        prob_map = {}
    elif isinstance(probs, pd.Series):
        prob_map = probs.to_dict()
    elif isinstance(probs, dict):
        prob_map = probs
    else:
        # list/ndarray -> align by candidates order
        try:
            prob_map = dict(zip(candidates.index, list(probs)))
        except Exception:
            logger.warning("Could not align probs array to candidates; defaulting to zeros.")
            prob_map = {}

    # iterate over candidates (use row index as fallback candidate_time)
    for idx, row in candidates.iterrows():
        candidate_time = row.get("candidate_time", idx)
        # normalize candidate_time to Timestamp if possible
        try:
            candidate_time = pd.to_datetime(candidate_time)
        except Exception:
            pass

        prob = float(prob_map.get(idx, prob_map.get(candidate_time, 0.0)))

        # assign layer based on prob thresholds
        if prob >= p_fast:
            layer = "fast"
        elif prob >= p_slow:
            layer = "shallow"
        elif prob >= p_deep:
            layer = "deep"
        else:
            # probability too low -> no fill
            continue

        # entry & exit times/prices
        entry_price = row.get("entry_price", np.nan)
        # prefer realized_return if provided
        realized_return = row.get("realized_return", None)
        if realized_return is None or (isinstance(realized_return, float) and np.isnan(realized_return)):
            # fallback: use a conservative small random return (to simulate slippage)
            mu = 0.001 if layer == "fast" else 0.0005 if layer == "shallow" else 0.0
            sigma = 0.005
            ret = float(np.random.normal(loc=mu, scale=sigma))
        else:
            ret = float(realized_return)

        size = float(row.get("size", 1.0) or 1.0)
        filled_at = candidate_time
        open_time = candidate_time
        close_time = row.get("end_time", candidate_time)

        # compute pnl
        pnl = size * ret
        win = bool(ret > 0)

        trades.append({
            "candidate_time": candidate_time,
            "open_time": open_time,
            "close_time": close_time,
            "filled_at": filled_at,
            "layer": layer,
            "entry_price": float(entry_price) if not pd.isna(entry_price) else None,
            "size": size,
            "ret": ret,
            "pnl": pnl,
            "win": win,
        })

    if not trades:
        # return empty typed dataframe
        cols = ["candidate_time","open_time","close_time","filled_at","layer","entry_price","size","ret","pnl","win"]
        return pd.DataFrame(columns=cols)

    trades_df = pd.DataFrame(trades)
    # ensure proper dtypes
    trades_df["candidate_time"] = pd.to_datetime(trades_df["candidate_time"])
    trades_df["open_time"] = pd.to_datetime(trades_df["open_time"])
    trades_df["close_time"] = pd.to_datetime(trades_df["close_time"])
    trades_df["filled_at"] = pd.to_datetime(trades_df["filled_at"])

    return trades_df


# -------------------------------------------------------------------------
def _compute_target_return_from_rr(row: pd.Series, rr: float) -> float:
    """
    Compute the target return for a candidate given RR and candidate fields.
    target_return = (rr * atr) / entry_price
    If atr or entry_price missing, return NaN.
    """
    atr = row.get("atr", None)
    entry = row.get("entry_price", None)
    try:
        if atr is None or entry is None or entry == 0:
            return float("nan")
        return float((rr * float(atr)) / float(entry))
    except Exception:
        return float("nan")


# -------------------------------------------------------------------------
def summarize_sweep(
    candidates: pd.DataFrame,
    rr_vals: Sequence[float],
    sl_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    mpt_list: Optional[Sequence[float]] = None,
    include_counts: bool = True,
) -> pd.DataFrame:
    """
    Quick summary for RR × model_prob_threshold grid.

    For each rr and mpt:
      - treat a candidate as a "win" for rr if realized_return >= target_return (rr × atr / entry)
      - if mpt_list is provided, interpret it as minimal model probability (but this function
        does not require a model — it will compute results assuming all candidates pass the prob).
      - returns aggregated stats.

    This is a fast, approximate sweep used for exploration and UI presentation.
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    rr_vals = list(rr_vals) if rr_vals is not None else [2.0]
    mpt_list = list(mpt_list) if mpt_list is not None and len(mpt_list) > 0 else [0.0]

    rows: List[Dict[str, Any]] = []
    for rr in rr_vals:
        # compute target returns vectorized
        tr = candidates.apply(lambda r: _compute_target_return_from_rr(r, rr), axis=1)
        # avoid NaNs by dropping
        valid_mask = ~tr.isna()
        if valid_mask.sum() == 0:
            continue
        realized = pd.to_numeric(candidates.loc[valid_mask, "realized_return"], errors="coerce").fillna(0.0)
        # wins per candidate for this rr
        wins = (realized >= tr.loc[valid_mask]).astype(int)

        for mpt in mpt_list:
            # In the absence of an actual model, assume all candidates are considered.
            # The UI will use model thresholds when a model is present. Here we just report
            # the underlying win rate for the rr selection.
            total = int(len(wins))
            wins_count = int(wins.sum())
            win_rate = float(wins_count / total) if total > 0 else 0.0
            mean_ret = float(realized.mean()) if total > 0 else 0.0
            median_ret = float(realized.median()) if total > 0 else 0.0
            std_ret = float(realized.std(ddof=0)) if total > 1 else 0.0

            rows.append({
                "rr": float(rr),
                "model_prob_threshold": float(mpt),
                "n_candidates": total,
                "n_wins": wins_count,
                "win_rate": win_rate,
                "mean_ret": mean_ret,
                "median_ret": median_ret,
                "std_ret": std_ret,
            })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
def run_backtest(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    rr_grid: Sequence[float],
    sl_grid: Optional[Sequence[Tuple[float, float]]] = None,
    session_modes: Optional[Sequence[str]] = None,
    model_prob_thresholds: Optional[Sequence[float]] = None,
    max_bars: int = 60,
    rvol_threshold: float = 1.5,
    train_on_allowed_session: bool = True,
    model_train_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Lightweight driver to run simple grid backtests.

    Returns a dict containing:
      - 'summary': DataFrame-like list of dicts (rr, model_prob_threshold, win_rate, n_trades, mean_ret, ...)
      - 'detailed_trades': dict keyed by 'rr|mpt' -> trades DataFrame

    NOTE: This function does not retrain models. It uses the candidates' realized_return to decide wins
    for different RR values and model probability thresholds. If you want per-grid-model retraining,
    call your training routine externally per cell and pass its probs into simulate_limits.
    """
    result = {"summary": [], "detailed_trades": {}}
    if candidates is None or candidates.empty:
        logger.warning("run_backtest: no candidates provided")
        return result

    rr_grid = list(rr_grid) if rr_grid is not None else [2.0]
    model_prob_thresholds = list(model_prob_thresholds) if model_prob_thresholds is not None and len(model_prob_thresholds) > 0 else [0.0]

    # Precompute a few values
    candidates = candidates.copy().reset_index(drop=True)
    candidates["_target_ret_by_rr"] = np.nan  # placeholder

    for rr in rr_grid:
        # compute target return for rr
        candidates["_target_ret_by_rr"] = candidates.apply(lambda r: _compute_target_return_from_rr(r, rr), axis=1)
        valid_mask = ~candidates["_target_ret_by_rr"].isna()
        if valid_mask.sum() == 0:
            logger.debug("run_backtest: rr=%s produced zero valid candidates", rr)
            continue

        # realized returns (for filtered set)
        realized = pd.to_numeric(candidates.loc[valid_mask, "realized_return"], errors="coerce").fillna(0.0)
        target_ret = candidates.loc[valid_mask, "_target_ret_by_rr"]

        # for each mpt, compute trades (here mpt acts as post-model filter but we assume all candidates pass)
        for mpt in model_prob_thresholds:
            # selection mask: in a full system you'd and with (model_proba >= mpt)
            sel_mask = valid_mask.copy()
            sel_idx = candidates.index[sel_mask]

            # wins
            wins = (realized >= target_ret).astype(int)
            total = int(len(wins))
            n_wins = int(wins.sum())
            win_rate = float(n_wins / total) if total > 0 else 0.0
            mean_ret = float(realized.mean()) if total > 0 else 0.0
            median_ret = float(realized.median()) if total > 0 else 0.0
            std_ret = float(realized.std(ddof=0)) if total > 1 else 0.0

            key = f"rr{rr}_mpt{mpt}"
            # Build trivial trades dataframe for this cell (one row per candidate selected)
            trades_rows = []
            for idx_sel in sel_idx:
                r = candidates.loc[idx_sel]
                ret = float(r.get("realized_return", 0.0))
                entry_price = r.get("entry_price", None)
                size = float(r.get("size", 1.0) or 1.0)
                hit = int(ret >= r.get("_target_ret_by_rr", np.nan)) if not np.isnan(r.get("_target_ret_by_rr", np.nan)) else 0
                trades_rows.append({
                    "candidate_time": r.get("candidate_time", pd.NaT),
                    "entry_price": entry_price,
                    "size": size,
                    "ret": ret,
                    "pnl": size * ret,
                    "win": bool(hit),
                    "rr": float(rr),
                    "model_prob_threshold": float(mpt),
                })
            trades_df = pd.DataFrame(trades_rows)
            result["detailed_trades"][key] = trades_df

            result["summary"].append({
                "rr": float(rr),
                "model_prob_threshold": float(mpt),
                "n_candidates": total,
                "n_wins": n_wins,
                "win_rate": win_rate,
                "mean_ret": mean_ret,
                "median_ret": median_ret,
                "std_ret": std_ret,
            })

    # convert summary to DataFrame-like structure for convenience downstream
    if result["summary"]:
        result["summary"] = pd.DataFrame(result["summary"])
    else:
        result["summary"] = pd.DataFrame()

    return result


# -------------------------------------------------------------------------
def run_breadth_backtest(
    candidates: pd.DataFrame,
    rr_vals: Optional[Sequence[float]] = None,
    sl_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    session_modes: Optional[Sequence[str]] = None,
    mpt_list: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Run Low / Mid / High breadth modes sequentially and return aggregated stats.

    Breadth modes (simple, rule-based):
      - Low Breadth:  prefer conservative rr in rr_vals (e.g., lower RR),
      - Mid Breadth:  choose medium RR,
      - High Breadth: prefer higher RR values (exploratory).

    The function returns a DataFrame with one row per breadth mode with stats:
      [mode, rr_used, n_candidates, n_wins, win_rate, mean_ret, median_ret, std_ret]

    Note: this is a rule-driven orchestration function for UI; it's not a full optimization engine.
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    rr_vals = sorted(list(rr_vals) if rr_vals is not None else [2.0])
    mpt_list = list(mpt_list) if mpt_list is not None and len(mpt_list) > 0 else [0.0]

    def _aggregate_for_rr(rr: float) -> Dict[str, Any]:
        tr = candidates.apply(lambda r: _compute_target_return_from_rr(r, rr), axis=1)
        valid_mask = ~tr.isna()
        if valid_mask.sum() == 0:
            return {"n_candidates": 0, "n_wins": 0, "win_rate": 0.0, "mean_ret": 0.0, "median_ret": 0.0, "std_ret": 0.0}
        realized = pd.to_numeric(candidates.loc[valid_mask, "realized_return"], errors="coerce").fillna(0.0)
        target_ret = tr.loc[valid_mask]
        wins = (realized >= target_ret).astype(int)
        total = int(len(wins))
        n_wins = int(wins.sum())
        win_rate = float(n_wins / total) if total > 0 else 0.0
        return {
            "n_candidates": total,
            "n_wins": n_wins,
            "win_rate": win_rate,
            "mean_ret": float(realized.mean()) if total > 0 else 0.0,
            "median_ret": float(realized.median()) if total > 0 else 0.0,
            "std_ret": float(realized.std(ddof=0)) if total > 1 else 0.0,
        }

    results: List[Dict[str, Any]] = []
    # Determine candidate counts per RR and pick RR for each breadth mode
    if len(rr_vals) == 1:
        low_rr = mid_rr = high_rr = rr_vals[0]
    else:
        low_rr = rr_vals[0]
        mid_rr = rr_vals[len(rr_vals) // 2]
        high_rr = rr_vals[-1]

    for mode, rr_choice in [("Low", low_rr), ("Mid", mid_rr), ("High", high_rr)]:
        stats = _aggregate_for_rr(rr_choice)
        row = {
            "mode": mode,
            "rr_used": float(rr_choice),
            **stats
        }
        results.append(row)

    return pd.DataFrame(results)
