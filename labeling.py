# labeling.py
import pandas as pd
import numpy as np

def generate_candidates_and_labels(df: pd.DataFrame, lookback: int = 64, k_tp: float = 2.0,
                                   k_sl: float = 1.0, atr_window: int = 20, max_bars: int = 60) -> pd.DataFrame:
    """
    Generate candidate entries and labels for confirm model training.
    Uses microstructure and ATR for labeling.
    """
    df = df.copy()
    # ATR calculation
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(window=atr_window, min_periods=1).mean()

    candidates = []
    for i in range(lookback, len(df)-max_bars):
        entry_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        future_high = df['high'].iloc[i+1:i+1+max_bars].max()
        future_low = df['low'].iloc[i+1:i+1+max_bars].min()

        label = 0
        if future_high >= entry_price + k_tp*atr:
            label = 1
        elif future_low <= entry_price - k_sl*atr:
            label = -1

        candidates.append({
            'candidate_time': df.index[i],
            'entry_price': entry_price,
            'atr': atr,
            'label': label,
            'tick_rate': np.random.rand(),       # placeholder feature
            'uptick_ratio': np.random.rand(),
            'buy_vol_ratio': np.random.rand(),
            'micro_range': np.random.rand(),
            'rvol_micro': np.random.rand(),
        })

    return pd.DataFrame(candidates)