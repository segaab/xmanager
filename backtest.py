import pandas as pd
import numpy as np

def simulate_limits(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    probs: pd.Series,
    p_fast: float = 0.7,
    p_slow: float = 0.55,
    p_deep: float = 0.45,
) -> pd.DataFrame:
    """
    Simulate fills based on confirm probabilities with three probability tiers.
    Uses the candidates' index as the event time (no longer requires a
    'candidate_time' column).
    """
    trades: list[dict] = []

    for idx, row in candidates.iterrows():          # idx == candidate_time
        prob = float(probs.get(idx, 0.0))

        # Decide layer
        if prob >= p_fast:
            layer = "fast"
        elif prob >= p_slow:
            layer = "shallow"
        elif prob >= p_deep:
            layer = "deep"
        else:
            continue  # skip – probability too low

        # Very simple return simulation (placeholder)
        ret = np.random.normal(loc=0.001, scale=0.005)
        size = 1.0

        trades.append(
            {
                "candidate_time": idx,
                "layer": layer,
                "entry_price": row["entry_price"],
                "size": size,
                "ret": ret,
                "pnl": size * ret,
                "filled_at": idx,
            }
        )

    return pd.DataFrame(trades)
