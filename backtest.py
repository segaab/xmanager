# backtest.py
import logging
from typing import Union, Optional

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

    Parameters:
        bars: pd.DataFrame of price bars (not used directly but for reference if needed)
        candidates: DataFrame including 'entry_price', optional 'realized_return', 'candidate_time', 'size'
        probs: pd.Series, dict, list, or np.ndarray aligned to candidates; defaults to 0
        p_fast, p_slow, p_deep: thresholds for assigning layers

    Returns:
        pd.DataFrame of simulated trades with columns:
            candidate_time, layer, entry_price, size, ret, pnl, filled_at
    """
    trades = []

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
        # Use candidate_time if exists, else index
        candidate_time = row.get("candidate_time", idx)
        prob = float(prob_map.get(idx, prob_map.get(candidate_time, 0.0)))

        if prob >= p_fast:
            layer = "fast"
        elif prob >= p_slow:
            layer = "shallow"
        elif prob >= p_deep:
            layer = "deep"
        else:
            continue  # skip low-prob events

        # Determine realized return
        realized_return = row.get("realized_return", None)
        if realized_return is None or (isinstance(realized_return, float) and np.isnan(realized_return)):
            # fallback small random return depending on layer
            mu = 0.001 if layer == "fast" else 0.0005 if layer == "shallow" else 0.0
            sigma = 0.005
            ret = float(np.random.normal(loc=mu, scale=sigma))
        else:
            ret = float(realized_return)

        size = float(row.get("size", 1.0) or 1.0)
        entry_price = row.get("entry_price", np.nan)
        entry_price = float(entry_price) if not pd.isna(entry_price) else None

        trades.append({
            "candidate_time": candidate_time,
            "layer": layer,
            "entry_price": entry_price,
            "size": size,
            "ret": ret,
            "pnl": size * ret,
            "filled_at": candidate_time,
        })

    if not trades:
        return pd.DataFrame(columns=["candidate_time", "layer", "entry_price", "size", "ret", "pnl", "filled_at"])

    return pd.DataFrame(trades)