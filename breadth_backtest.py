# breadth_backtest.py
"""
Breadth backtest runner.

Provides `run_breadth_backtest(clean, rr_vals, sl_ranges, session_modes, mpt_list)` which
computes backtest summaries across three breadth regimes (Low, Mid, High) defined by
per-day candidate density percentiles.

The function returns a pd.DataFrame summarizing results per breadth-mode × rr × mpt.
It also returns a dict of detailed trades keyed by 'mode__rr__mpt' when requested.

Notes
-----
This is a pragmatic, reproducible implementation intended for exploratory analysis.
"""

from typing import Iterable, Tuple, List, Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
from backtest import summarize_sweep  # uses the summarize_sweep implementation above

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _session_groups_by_density(clean: pd.DataFrame, low_pct: float = 0.2, high_pct: float = 0.8):
    """
    Compute per-day candidate counts and split days into low/mid/high breadth groups.
    Returns dict: { 'low': [dates], 'mid': [dates], 'high': [dates] }
    """
    if "candidate_time" not in clean.columns:
        clean = clean.copy()
        clean["candidate_time"] = pd.to_datetime(clean.index)

    clean["_cand_date"] = pd.to_datetime(clean["candidate_time"]).dt.normalize()
    daily_counts = clean.groupby("_cand_date").size().rename("count")
    if daily_counts.empty:
        return {"low": [], "mid": [], "high": []}

    low_th = daily_counts.quantile(low_pct)
    high_th = daily_counts.quantile(high_pct)

    low_days = daily_counts[daily_counts <= low_th].index.tolist()
    mid_days = daily_counts[(daily_counts > low_th) & (daily_counts <= high_th)].index.tolist()
    high_days = daily_counts[daily_counts > high_th].index.tolist()

    return {"low": low_days, "mid": mid_days, "high": high_days}


def _filter_by_session_mode(clean: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Apply session_mode filtering. Modes:
      - 'all' : return input
      - 'top_k:N' : keep candidates from top-N active days (most candidates)
    """
    if clean is None or clean.empty:
        return clean

    if "candidate_time" not in clean.columns:
        clean = clean.copy()
        clean["candidate_time"] = pd.to_datetime(clean.index)

    clean_local = clean.copy()
    clean_local["_cand_date"] = pd.to_datetime(clean_local["candidate_time"]).dt.normalize()
    daily_counts = clean_local.groupby("_cand_date").size().rename("count")
    daily_counts = daily_counts.sort_values(ascending=False)

    if mode == "all":
        return clean_local.drop(columns=["_cand_date"])

    if mode.startswith("top_k"):
        try:
            k = int(mode.split(":")[1])
        except Exception:
            k = 3
        topk_dates = daily_counts.head(k).index.tolist()
        out = clean_local[clean_local["_cand_date"].isin(topk_dates)].drop(columns=["_cand_date"])
        return out

    # unknown mode -> return all
    return clean_local.drop(columns=["_cand_date"])


def run_breadth_backtest(
    clean: pd.DataFrame,
    rr_vals: Iterable[float],
    sl_ranges: Iterable[Tuple[float, float]],
    session_modes: Iterable[str],
    mpt_list: Iterable[float],
    return_detailed: bool = True,
) -> Dict[str, Any]:
    """
    Run breadth backtests for Low / Mid / High breadth regimes.

    Parameters
    ----------
    clean : pd.DataFrame
        Candidates DataFrame (must contain candidate_time, entry_price, atr, realized_return)
    rr_vals, sl_ranges, mpt_list : iterables
        Grids to evaluate (see summarize_sweep)
    session_modes : iterable of str
        Modes controlling session selection, e.g., ["all", "top_k:3"]
    return_detailed : bool
        If True, returns a dict of detailed trades keyed by 'mode__rr__mpt'.

    Returns
    -------
    dict with:
      - summary_df : pd.DataFrame  (mode, session_mode, rr, sl_min, sl_max, model_prob_threshold, num_trades, win_rate, avg_ret, total_pnl)
      - detailed : dict (optional) mapping keys -> DataFrame
    """
    if clean is None or clean.empty:
        return {"summary_df": pd.DataFrame(), "detailed": {}}

    # compute breadth buckets by per-day density
    buckets = _session_groups_by_density(clean)
    if not any(len(v) for v in buckets.values()):
        # fallback: treat all as 'mid'
        buckets = {"low": [], "mid": list(pd.to_datetime(clean['candidate_time']).dt.normalize().unique()), "high": []}

    all_rows = []
    detailed_map: Dict[str, pd.DataFrame] = {}

    for breadth_mode, days in buckets.items():
        # If days empty, we still want to allow "all" via session_modes filtering
        for session_mode in session_modes:
            # first apply session_mode filter
            filtered = _filter_by_session_mode(clean, session_mode)

            # then, if the breadth bucket has days, restrict to those dates
            if days:
                filtered_dates = pd.to_datetime(filtered['candidate_time']).dt.normalize()
                mask = filtered_dates.isin(days)
                filtered = filtered.loc[mask].copy()

            if filtered.empty:
                logger.info("No candidates for breadth=%s session_mode=%s", breadth_mode, session_mode)
                # still append zero-rows for the grid cells
                for rr in rr_vals:
                    for sl in sl_ranges:
                        for mpt in mpt_list:
                            all_rows.append({
                                "breadth_mode": breadth_mode,
                                "session_mode": session_mode,
                                "rr": float(rr),
                                "sl_min": float(sl[0]),
                                "sl_max": float(sl[1]),
                                "model_prob_threshold": float(mpt),
                                "num_trades": 0,
                                "win_rate": np.nan,
                                "avg_ret": np.nan,
                                "total_pnl": 0.0,
                            })
                continue

            # call summarize_sweep on this filtered set to get the grid results (it returns a DataFrame in 'summary')
            sweep_res = summarize_sweep(filtered, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list)
            summary_df = sweep_res.get("summary", pd.DataFrame())
            detailed = sweep_res.get("detailed_trades", {})

            # annotate rows and collect
            for _, r in summary_df.iterrows():
                all_rows.append({
                    "breadth_mode": breadth_mode,
                    "session_mode": session_mode,
                    "rr": float(r["rr"]),
                    "sl_min": float(r["sl_min"]),
                    "sl_max": float(r["sl_max"]),
                    "model_prob_threshold": float(r["model_prob_threshold"]),
                    "num_trades": int(r["num_trades"]),
                    "win_rate": float(r["win_rate"]) if not np.isnan(r["win_rate"]) else np.nan,
                    "avg_ret": float(r["avg_ret"]) if not np.isnan(r["avg_ret"]) else np.nan,
                    "total_pnl": float(r["total_pnl"]),
                })

            if return_detailed:
                # attach detailed trades but key them by breadth/session/rr/mpt
                for k, df_tr in (detailed or {}).items():
                    # key k formatted like rr{rr}_sl{min}-{max}_mpt{mpt}
                    new_key = f"{breadth_mode}__{session_mode}__{k}"
                    detailed_map[new_key] = df_tr

    out_df = pd.DataFrame(all_rows)
    return {"summary_df": out_df, "detailed": detailed_map}
