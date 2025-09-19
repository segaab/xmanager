# backtest.py
import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_limits(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    probs: Union[pd.Series, dict, list, np.ndarray] = None,
    p_fast: float = 0.7,
    p_slow: float = 0.55,
    p_deep: float = 0.45,
) -> pd.DataFrame:
    """
    Simulate fills based on confirm probabilities and candidate realized returns.

    - `candidates` expected to have at least ['candidate_time','entry_price'] or use index as time,
      and optionally 'realized_return' column computed by labeling function.
    - `probs` can be a pd.Series indexed by candidate index, a dict mapping index->prob,
      or an array aligned to candidates.index order.
    """
    trades = []

    # normalize probs to a lookup dict
    prob_map = {}
    if probs is None:
        prob_map = {}
    elif isinstance(probs, pd.Series):
        prob_map = probs.to_dict()
    elif isinstance(probs, dict):
        prob_map = probs
    else:
        # assume array-like aligned with candidates.index order
        try:
            prob_map = dict(zip(candidates.index, list(probs)))
        except Exception:
            logger.warning("Could not align probs array to candidates; defaulting to zero probs.")
            prob_map = {}

    for row in candidates.itertuples(index=True):
        # support both index-as-time or explicit candidate_time column
        idx = getattr(row, "Index", None)
        candidate_time = getattr(row, "candidate_time", None)
        event_key = idx if candidate_time is None else candidate_time

        prob = float(prob_map.get(idx, prob_map.get(candidate_time, 0.0)))

        if prob >= p_fast:
            layer = "fast"
        elif prob >= p_slow:
            layer = "shallow"
        elif prob >= p_deep:
            layer = "deep"
        else:
            continue

        # use realized_return if available for deterministic backtest
        realized_return = getattr(row, "realized_return", None)
        if realized_return is None or (isinstance(realized_return, float) and np.isnan(realized_return)):
            # fallback: small random return proportional to layer
            mu = 0.001 if layer == "fast" else 0.0005 if layer == "shallow" else 0.0
            sigma = 0.005
            ret = float(np.random.normal(loc=mu, scale=sigma))
        else:
            ret = float(realized_return)

        size = 1.0
        entry_price = getattr(row, "entry_price", np.nan)

        trades.append({
            "candidate_time": candidate_time if candidate_time is not None else idx,
            "layer": layer,
            "entry_price": float(entry_price) if not pd.isna(entry_price) else None,
            "size": float(size),
            "ret": float(ret),
            "pnl": float(size * ret),
            "filled_at": candidate_time if candidate_time is not None else idx,
        })

    if not trades:
        return pd.DataFrame(columns=["candidate_time","layer","entry_price","size","ret","pnl","filled_at"])

    return pd.DataFrame(trades)