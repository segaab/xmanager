# labeling.py
import numpy as np
import pandas as pd

def triple_barrier_label(bars: pd.DataFrame, entry_index, entry_price,
                         k_tp=2.0, k_sl=1.0, atr_window=20, max_bars=60):
    """
    Simple triple-barrier that works on bar-index level (minute/daily).
    bars: DataFrame with 'high','low','close' indexed by datetime
    entry_index: index location (integer) where entry occurs
    entry_price: float
    returns: (outcome, exit_index, exit_price, return)
       outcome: +1 tp hit first, -1 sl hit first, 0 vertical
    """
    # compute ATR
    high = bars['high']
    low = bars['low']
    close = bars['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=1).mean().fillna(method='bfill')

    if entry_index >= len(bars)-1:
        return 0, entry_index, entry_price, 0.0

    tp = entry_price + k_tp * atr.iloc[entry_index]
    sl = entry_price - k_sl * atr.iloc[entry_index]
    # search next max_bars bars
    for i in range(entry_index+1, min(len(bars), entry_index + max_bars + 1)):
        if bars['high'].iloc[i] >= tp:
            ret = (tp - entry_price)/entry_price
            return 1, i, tp, ret
        if bars['low'].iloc[i] <= sl:
            ret = (sl - entry_price)/entry_price
            return -1, i, sl, ret
    # vertical
    final_price = bars['close'].iloc[min(len(bars)-1, entry_index + max_bars)]
    ret = (final_price - entry_price)/entry_price
    return 0, entry_index + max_bars, final_price, ret

def generate_candidates_and_labels(bars: pd.DataFrame, lookback=64, step_ticks=1,
                                   k_tp=2.0, k_sl=1.0, atr_window=20, max_bars=60):
    """
    For demonstration: for each bar after lookback create a candidate at midpoint of
    recent range and run triple barrier to produce a label (good fill or not).
    Returns DataFrame with features and label.
    """
    rows = []
    for t in range(lookback, len(bars)-max_bars):
        window = bars.iloc[t-lookback:t]
        # simple entry band: last bar close +/- small percent
        last_close = bars['close'].iloc[t]
        band_low = last_close * 0.997
        band_high = last_close * 1.003
        entry_price = (band_low + band_high) / 2.0
        outcome, exit_i, exit_p, ret = triple_barrier_label(bars, t, entry_price,
                                                            k_tp=k_tp, k_sl=k_sl,
                                                            atr_window=atr_window, max_bars=max_bars)
        # compute short window micro features using minute bars as proxy
        micro = bars.iloc[max(0, t-30):t+1]
        tick_rate = len(micro) / (30)  # per-second proxy is rough
        upticks = (micro['close'].diff() > 0).sum()
        uptick_ratio = upticks / max(1, len(micro))
        buy_vol_ratio = micro['volume'][micro['close'].diff() > 0].sum() / (micro['volume'].sum() + 1e-9)
        micro_range = micro['high'].max() - micro['low'].min()
        rvol = micro['volume'].sum() / (micro['volume'].rolling(20).mean().iloc[-1] + 1e-9)
        rows.append({
            't_idx': t,
            'datetime': bars.index[t],
            'entry_price': entry_price,
            'tick_rate': tick_rate,
            'uptick_ratio': uptick_ratio,
            'buy_vol_ratio': buy_vol_ratio,
            'micro_range': micro_range,
            'rvol_micro': rvol,
            'label': 1 if outcome==1 else 0,  # success=TP first
            'ret': ret
        })
    df = pd.DataFrame(rows).set_index('datetime')
    return df