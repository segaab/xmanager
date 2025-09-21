# sweep.py
"""
Sweep evaluation utilities for candidate signals.

Provides functions to evaluate a grid of risk-reward / stop-loss / model probability thresholds
and compute summary statistics and detailed trade DataFrames.

Intended to be used for:
    - grid sweeps
    - parameter optimization
    - pre-backtest filtering

Dependencies
------------
- pandas, numpy, logging
- backtest.py functions (simulate_limits, summarize_sweep)
"""

from typing import Iterable, Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from backtest import simulate_limits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def sweep_candidates(
    candidates: pd.DataFrame,
    rr_vals: Iterable[float],
    sl_ranges: Iterable[Tuple[float, float]],
    mpt_list: Iterable[float],
    assume_direction: str = "long",
) -> Dict[str, Any]:
    """
    Evaluate a sweep grid for candidate entries.

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate signals. Must include:
            - candidate_time (datetime)
            - entry_price (float)
            - atr (float)
            - realized_return (float)
            - optional: pred_prob / confirm_proba
    rr_vals : iterable of float
        Risk-Reward multipliers
    sl_ranges : iterable of (float, float)
        Stop-loss ranges (expressed in ATR multiples)
    mpt_list : iterable of float
        Minimum probability thresholds for model confirmation
    assume_direction : str
        "long" or "short" (used for interpreting return thresholds)

    Returns
    -------
    dict with:
        - summary : pd.DataFrame summarizing grid results
        - detailed_trades : dict mapping grid-cell keys -> DataFrames
    """
    if candidates is None or candidates.empty:
        return {"summary": pd.DataFrame(), "detailed_trades": {}}

    # run simulate_limits optionally if needed for fills
    # currently we assume realized_return is available
    # candidates_filled = simulate_limits(candidates)

    # compute summary per grid cell
    from backtest import summarize_sweep

    sweep_res = summarize_sweep(
        clean=candidates,
        rr_vals=rr_vals,
        sl_ranges=sl_ranges,
        mpt_list=mpt_list,
        assume_direction=assume_direction,
    )
    return sweep_res


def best_grid_cell(summary: pd.DataFrame, metric: str = "total_pnl", top_n: int = 1) -> pd.DataFrame:
    """
    Select the best performing grid cells according to a metric.

    Parameters
    ----------
    summary : pd.DataFrame
        Output from sweep_candidates()["summary"]
    metric : str
        Column name to sort by (e.g., "total_pnl", "avg_ret", "win_rate")
    top_n : int
        Number of top-performing cells to return

    Returns
    -------
    pd.DataFrame subset of summary with top_n rows
    """
    if summary is None or summary.empty or metric not in summary.columns:
        return pd.DataFrame()
    return summary.sort_values(by=metric, ascending=False).head(top_n).reset_index(drop=True)


def filter_candidates_by_prob(
    candidates: pd.DataFrame, min_prob: float = 0.5, prob_col: str = "pred_prob"
) -> pd.DataFrame:
    """
    Filter candidate signals by minimum probability.

    Parameters
    ----------
    candidates : pd.DataFrame
    min_prob : float
        Minimum probability threshold
    prob_col : str
        Column containing model probability

    Returns
    -------
    pd.DataFrame
        Subset of candidates exceeding min_prob
    """
    if candidates is None or candidates.empty or prob_col not in candidates.columns:
        return candidates
    return candidates[candidates[prob_col].astype(float) >= min_prob].copy()