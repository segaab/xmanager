# backtest.py
import pandas as pd
import numpy as np

def simulate_limits(bars: pd.DataFrame, candidates: pd.DataFrame, probs: pd.Series,
                    p_fast: float = 0.7, p_slow: float = 0.55, p_deep: float = 0.45) -> pd.DataFrame:
    """
    Simulate fills based on confirm probabilities with thresholds.
    """
    trades = []
    for i, row in candidates.iterrows():
        prob = probs.get(row['candidate_time'], 0.0)
        take_trade = False
        if prob >= p_fast:
            take_trade = True
            layer = 'fast'
        elif prob >= p_slow:
            take_trade = True
            layer = 'shallow'
        elif prob >= p_deep:
            take_trade = True
            layer = 'deep'

        if take_trade:
            # Simulate return as small random fraction
            ret = np.random.normal(loc=0.001, scale=0.005)
            size = 1.0
            trades.append({
                'candidate_time': row['candidate_time'],
                'layer': layer,
                'entry_price': row['entry_price'],
                'size': size,
                'ret': ret,
                'pnl': size*ret,
                'filled_at': row['candidate_time']
            })

    return pd.DataFrame(trades)