# backtest.py
import numpy as np
import pandas as pd

def simulate_limits(bars: pd.DataFrame, candidates: pd.DataFrame, confirm_probs: np.ndarray,
                    p_fast=0.7, p_slow=0.55, p_deep=0.45, layered_sizes={'ideal':0.5,'shallow':0.3,'deep':0.2},
                    W_fill_bars=5, RR_min=1.5):
    """
    For each candidate (row aligned to a bar index), decide which layers to post and simulate fills.
    Uses price-touch logic + confirm_prob threshold; no queue-depth modeling.
    Returns list of simulated trades with realized returns and sizes.
    """
    trades = []
    for i, (idx, row) in enumerate(candidates.iterrows()):
        prob = confirm_probs[i]
        entry_price = row['entry_price']
        # determine postings
        posted = []
        if prob >= p_fast:
            posted.append(('ideal', layered_sizes['ideal'], entry_price))
        if prob >= p_slow:
            posted.append(('shallow', layered_sizes['shallow'], entry_price * 0.999))  # slightly worse price
        if prob >= p_deep:
            posted.append(('deep', layered_sizes['deep'], entry_price * 0.998))
        # simulate lookahead
        start_idx = row['t_idx']
        end_idx = start_idx + W_fill_bars
        window = bars.iloc[start_idx+1:end_idx+1]
        for layer_name, size, price_post in posted:
            filled = False
            fill_time = None
            fill_price = None
            for j, tick in enumerate(window.itertuples()):
                # simple touch logic: if high >= price_post => filled (for buys)
                if tick.high >= price_post:
                    filled = True
                    fill_time = window.index[j]
                    fill_price = price_post
                    break
            if not filled:
                continue
            # if filled, run triple-barrier from fill location (approx)
            # here we simplify: use return from labeling DF 'ret' if available
            realized_ret = row.get('ret', None)
            trades.append({
                'candidate_time': idx,
                'layer': layer_name,
                'size': size,
                'entry_price': fill_price,
                'filled_at': fill_time,
                'ret': realized_ret
            })
    trades_df = pd.DataFrame(trades)
    return trades_df