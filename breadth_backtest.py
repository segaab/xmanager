# breadth_backtest.py
"""
Thin wrapper module providing run_breadth_backtest used by the Streamlit UI.

This file exists to keep the app import stable (app.py imports run_breadth_backtest
from breadth_backtest). The heavy lifting is delegated to backtest.run_breadth_backtest,
but this wrapper adds argument normalization, simple validation, and friendly logging.

Exports
-------
run_breadth_backtest(candidates, rr_vals=None, sl_ranges=None, session_modes=None, mpt_list=None)
    Returns a pandas.DataFrame with one row per breadth mode ("Low","Mid","High").
"""

from __future__ import annotations
import logging
from typing import Optional, Sequence, Tuple, Any
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    # import the real implementation from backtest.py (should exist in same directory)
    from backtest import run_breadth_backtest as _core_run_breadth
except Exception as exc:
    logger.warning("Could not import backtest.run_breadth_backtest: %s", exc)
    _core_run_breadth = None


def run_breadth_backtest(
    candidates: pd.DataFrame,
    rr_vals: Optional[Sequence[float]] = None,
    sl_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    session_modes: Optional[Sequence[str]] = None,
    mpt_list: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Run Low/Mid/High breadth backtests and return aggregated statistics DataFrame.

    This wrapper normalizes inputs and delegates to backtest.run_breadth_backtest.
    If the core implementation is not available, returns an informative empty DataFrame.

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate events produced by labeling.generate_candidates_and_labels.
    rr_vals : sequence of floats, optional
        List/sequence of RR values to consider (e.g., [1.5,2.0,3.0]).
    sl_ranges : sequence of (float,float), optional
        SL percentage ranges — currently accepted for API compatibility (not used by simple breadth).
    session_modes : sequence of str, optional
        Session mode hints (e.g., ["all","top_k:3"]). Passed through to core if supported.
    mpt_list : sequence of float, optional
        Model probability thresholds (for completeness).

    Returns
    -------
    pd.DataFrame
        One-row-per-breadth-mode summary with columns:
        ['mode','rr_used','n_candidates','n_wins','win_rate','mean_ret','median_ret','std_ret']
    """
    if candidates is None or not isinstance(candidates, pd.DataFrame) or candidates.empty:
        logger.warning("run_breadth_backtest called with empty or invalid candidates DataFrame.")
        return pd.DataFrame(
            columns=["mode","rr_used","n_candidates","n_wins","win_rate","mean_ret","median_ret","std_ret"]
        )

    # Default rr_vals if none provided
    rr_vals = list(rr_vals) if rr_vals is not None and len(rr_vals) > 0 else [2.0, 3.0]
    mpt_list = list(mpt_list) if mpt_list is not None and len(mpt_list) > 0 else [0.0]

    # If the core implementation is available, delegate.
    if _core_run_breadth is not None:
        try:
            df = _core_run_breadth(
                candidates=candidates,
                rr_vals=rr_vals,
                sl_ranges=sl_ranges,
                session_modes=session_modes,
                mpt_list=mpt_list,
            )
            if not isinstance(df, pd.DataFrame):
                # core returned something unexpected — convert if possible
                try:
                    df = pd.DataFrame(df)
                except Exception:
                    logger.warning("Core run_breadth_backtest returned non-DataFrame; returning empty DataFrame.")
                    return pd.DataFrame(
                        columns=["mode","rr_used","n_candidates","n_wins","win_rate","mean_ret","median_ret","std_ret"]
                    )
            return df
        except Exception as exc:
            logger.exception("Error while executing core run_breadth_backtest: %s", exc)
            # fall through to a safe fallback

    # Fallback simple heuristics if core not available or failed:
    logger.info("Using fallback breadth computation (core implementation missing).")

    rr_vals_sorted = sorted(rr_vals)
    if len(rr_vals_sorted) == 1:
        low_rr = mid_rr = high_rr = rr_vals_sorted[0]
    else:
        low_rr = rr_vals_sorted[0]
        mid_rr = rr_vals_sorted[len(rr_vals_sorted) // 2]
        high_rr = rr_vals_sorted[-1]

    def _quick_stats_for_rr(rr: float):
        # compute implied target returns: target = rr * atr / entry_price
        def _target(row):
            try:
                atr = float(row.get("atr", float("nan")))
                entry = float(row.get("entry_price", float("nan")))
                if pd.isna(atr) or pd.isna(entry) or entry == 0:
                    return float("nan")
                return (rr * atr) / entry
            except Exception:
                return float("nan")

        targets = candidates.apply(_target, axis=1)
        mask = ~targets.isna()
        if mask.sum() == 0:
            return {"n_candidates": 0, "n_wins": 0, "win_rate": 0.0, "mean_ret": 0.0, "median_ret": 0.0, "std_ret": 0.0}
        realized = pd.to_numeric(candidates.loc[mask, "realized_return"], errors="coerce").fillna(0.0)
        wins = (realized >= targets.loc[mask]).astype(int)
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

    rows = []
    for mode, rr_choice in [("Low", low_rr), ("Mid", mid_rr), ("High", high_rr)]:
        stats = _quick_stats_for_rr(rr_choice)
        row = {"mode": mode, "rr_used": float(rr_choice)}
        row.update(stats)
        rows.append(row)

    return pd.DataFrame(rows)
