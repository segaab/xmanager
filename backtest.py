# backtest.py
"""
Backtest utilities: simulate_limits (fills) + summarize_sweep for grid sweeps.

Functions
---------
- simulate_limits(bars, candidates, probs, p_fast, p_slow, p_deep)
    Simulate fills based on confirm probabilities and candidate realized returns.

- summarize_sweep(clean, rr_vals, sl_ranges, mpt_list, assume_direction='long')
    Lightweight grid-sweep summarizer that computes metrics for combinations of
    RR / SL-range / model-probability-threshold using available candidate fields.

Notes
-----
This module is intentionally conservative: it expects the `clean` candidates DataFrame
to contain (at least) these columns:
    - candidate_time (datetime-like)
    - entry_price (float)
    - atr (float)
    - realized_return (float)    # return actually observed over candidate horizon
    - label (0/1)
Optionally:
    - pred_prob / pred_proba / confirm_proba (used to filter trades by probability)
If no probability column exists a default of 0.0 is assumed (you can set it upstream).
"""
from __future__ import annotations
from typing import Union, Optional, Iterable, Tuple, List, Dict, Any
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def simulate_limits(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    probs: Union[pd.Series, dict, list, np.ndarray, None] = None,
    p_fast: float = 0.7,
    p_slow: float = 0.55,
    p_deep: float = 0.45,
) -> pd.DataFrame:
    """
    Simulate fills based on confirm probabilities and candidate realized returns.

    Parameters
    ----------
    bars : pd.DataFrame
        Price bars (indexed by timestamp). Not required but used for plotting/reference.
    candidates : pd.DataFrame
        Candidate rows including entry_price, realized_return and candidate_time.
        Index may be arbitrary; this function will use candidates.index and
        'candidate_time' field if present.
    probs : pd.Series | dict | list | np.ndarray | None
        Probabilities aligned to candidates (by index or by candidate_time).
    p_fast, p_slow, p_deep : float
        Thresholds for layering.

    Returns
    -------
    pd.DataFrame of trades with columns:
        candidate_time, layer, entry_price, size, ret, pnl, filled_at
    """
    trades: List[Dict[str, Any]] = []

    # Build probability mapping
    prob_map = {}
    if probs is None:
        prob_map = {}
    elif isinstance(probs, pd.Series):
        prob_map = probs.to_dict()
    elif isinstance(probs, dict):
        prob_map = probs
    else:
        try:
            prob_map = dict(zip(candidates.index, list(probs)))
        except Exception:
            logger.warning("Could not align probs array to candidates; defaulting to zeros.")
            prob_map = {}

    for idx, row in candidates.iterrows():
        candidate_time = row.get("candidate_time", idx)
        prob = float(prob_map.get(idx, prob_map.get(candidate_time, 0.0)))

        # decide whether to attempt fill
        if prob >= p_fast:
            layer = "fast"
        elif prob >= p_slow:
            layer = "shallow"
        elif prob >= p_deep:
            layer = "deep"
        else:
            continue

        # realized return if available, else fallback random small return by layer
        realized_return = row.get("realized_return", None)
        if realized_return is None or (isinstance(realized_return, float) and np.isnan(realized_return)):
            mu = 0.001 if layer == "fast" else 0.0005 if layer == "shallow" else 0.0
            sigma = 0.005
            ret = float(np.random.normal(loc=mu, scale=sigma))
        else:
            ret = float(realized_return)

        size = float(row.get("size", 1.0) or 1.0)
        entry_price = row.get("entry_price", np.nan)
        entry_price = float(entry_price) if not pd.isna(entry_price) else None

        trades.append(
            {
                "candidate_time": candidate_time,
                "layer": layer,
                "entry_price": entry_price,
                "size": size,
                "ret": ret,
                "pnl": size * ret,
                "filled_at": candidate_time,
            }
        )

    if not trades:
        return pd.DataFrame(columns=["candidate_time", "layer", "entry_price", "size", "ret", "pnl", "filled_at"])
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------
# Sweep summarizer
# ---------------------------------------------------------------------
def _get_prob_series(clean: pd.DataFrame) -> pd.Series:
    """Pick the best available probability column or return zeros."""
    for c in ["pred_prob", "pred_proba", "confirm_proba", "proba", "prob"]:
        if c in clean.columns:
            return pd.to_numeric(clean[c], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=clean.index)


def summarize_sweep(
    clean: pd.DataFrame,
    rr_vals: Iterable[float],
    sl_ranges: Iterable[Tuple[float, float]],
    mpt_list: Iterable[float],
    assume_direction: str = "long",
) -> Dict[str, Any]:
    """
    Lightweight sweep summary.

    Parameters
    ----------
    clean : pd.DataFrame
        Clean candidate DataFrame (must contain entry_price, atr, realized_return, candidate_time).
    rr_vals : iterable of floats
        Risk-Reward multipliers to evaluate.
    sl_ranges : iterable of (sl_min, sl_max) tuples
        Stop-loss ranges (expressed in multiples of ATR).
        Each tuple is interpreted as (sl_min, sl_max); we use the midpoint as representative.
    mpt_list : iterable of floats
        Model probability thresholds (i.e., minimum prob to attempt the entry).

    Returns
    -------
    dict with keys:
        - summary: list of dicts (rr, sl_range, model_prob_threshold, num_trades, win_rate, avg_ret, total_pnl)
        - detailed_trades: dict mapping 'rr__sl__mpt' -> pd.DataFrame of simulated trades (subset)
    """
    if clean is None or clean.empty:
        return {"summary": [], "detailed_trades": {}}

    # ensure time column
    if "candidate_time" not in clean.columns:
        clean = clean.copy()
        clean["candidate_time"] = pd.to_datetime(clean.index)

    probs = _get_prob_series(clean)

    summary_rows = []
    detailed: Dict[str, pd.DataFrame] = {}

    # we require atr and entry_price to compute threshold returns; fall back gracefully
    atr_arr = pd.to_numeric(clean.get("atr", pd.Series(np.nan, index=clean.index)), errors="coerce").fillna(0.0)
    entry_arr = pd.to_numeric(clean.get("entry_price", pd.Series(np.nan, index=clean.index)), errors="coerce").fillna(np.nan)
    realized = pd.to_numeric(clean.get("realized_return", pd.Series(np.nan, index=clean.index)), errors="coerce")
    direction = assume_direction.lower()

    for rr in rr_vals:
        for sl_range in sl_ranges:
            sl_min, sl_max = float(sl_range[0]), float(sl_range[1])
            sl_mid = (sl_min + sl_max) / 2.0

            # compute thresholds in return-space (approx)
            # rr_return = (rr * atr) / entry_price
            # sl_return = - (sl_pct * atr) / entry_price
            rr_return = (rr * atr_arr) / (entry_arr.replace(0, np.nan))
            sl_return = - (sl_mid * atr_arr) / (entry_arr.replace(0, np.nan))

            # replace inf/nan with large sentinel so conditions are false
            rr_return = rr_return.replace([np.inf, -np.inf], np.nan).fillna(9e9)
            sl_return = sl_return.replace([np.inf, -np.inf], np.nan).fillna(-9e9)

            for mpt in mpt_list:
                # select candidate indices that would be attempted by model threshold
                mask_attempt = probs >= float(mpt)
                selected_idx = clean.index[mask_attempt]

                # if none selected, record zeros
                if len(selected_idx) == 0:
                    row = {
                        "rr": float(rr),
                        "sl_min": float(sl_min),
                        "sl_max": float(sl_max),
                        "sl_mid": float(sl_mid),
                        "model_prob_threshold": float(mpt),
                        "num_trades": 0,
                        "win_rate": np.nan,
                        "avg_ret": np.nan,
                        "total_pnl": 0.0,
                    }
                    summary_rows.append(row)
                    detailed_key = f"rr{rr}_sl{sl_min}-{sl_max}_mpt{mpt}"
                    detailed[detailed_key] = pd.DataFrame()
                    continue

                sel_realized = realized.loc[selected_idx].astype(float)
                sel_rr_ret = rr_return.loc[selected_idx].astype(float)
                sel_sl_ret = sl_return.loc[selected_idx].astype(float)

                # Determine wins/losses approximately: if realized_return >= rr_return => win
                wins_mask = sel_realized >= sel_rr_ret
                # losses: realized_return <= sl_return
                losses_mask = sel_realized <= sel_sl_ret
                # remaining (vertical barrier/no hit) considered losses (conservative)
                others_mask = ~(wins_mask | losses_mask)

                wins = wins_mask.sum()
                losses = losses_mask.sum() + others_mask.sum()
                num = len(selected_idx)
                win_rate = float(wins) / num if num > 0 else 0.0
                avg_ret = float(sel_realized.mean())
                total_pnl = float(sel_realized.sum())

                row = {
                    "rr": float(rr),
                    "sl_min": float(sl_min),
                    "sl_max": float(sl_max),
                    "sl_mid": float(sl_mid),
                    "model_prob_threshold": float(mpt),
                    "num_trades": int(num),
                    "win_rate": float(win_rate),
                    "avg_ret": float(avg_ret),
                    "total_pnl": float(total_pnl),
                }
                summary_rows.append(row)

                # produce a detailed trades frame for this grid cell
                df_sel = clean.loc[selected_idx].copy()
                df_sel["_win_rule"] = np.where(wins_mask.loc[selected_idx], "tp", np.where(losses_mask.loc[selected_idx], "sl", "other"))
                detailed_key = f"rr{rr}_sl{sl_min}-{sl_max}_mpt{mpt}"
                detailed[detailed_key] = df_sel.reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows)
    return {"summary": summary_df, "detailed_trades": detailed}