# labeling.py  (replace your existing function with this version)
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

def _compute_atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    high = bars['high']
    low = bars['low']
    close = bars['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    rvol_threshold: float = 1.5,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Compatibility wrapper that uses SL = k_sl * ATR and TP = k_tp * ATR.
    Default keeps your requested 3R behavior (k_tp=3, k_sl=1).
    """
    # Basic sanity checks
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must be indexed by a pandas.DatetimeIndex")

    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(set(bars.columns)):
        raise ValueError(f"`bars` must contain columns: {required}")

    df = bars.copy().sort_index()

    # compute rvol if missing
    if 'rvol' not in df.columns:
        vol_mean = df['volume'].rolling(window=20, min_periods=1).mean()
        df['rvol'] = df['volume'] / (vol_mean.replace(0, np.nan))

    # compute ATR
    atr = _compute_atr(df, window=atr_window)
    df['atr'] = atr

    cand_mask = (df['rvol'] >= rvol_threshold) & (np.arange(len(df)) >= lookback)
    candidate_times = df.index[cand_mask]

    records = []
    for t in candidate_times:
        entry_idx = df.index.get_loc(t)
        entry_price = float(df['close'].iat[entry_idx])
        atr_t = float(df['atr'].iat[entry_idx]) if not pd.isna(df['atr'].iat[entry_idx]) else np.nan
        if not np.isfinite(atr_t) or atr_t <= 0:
            continue

        sl_size = k_sl * atr_t
        tp_size = k_tp * atr_t

        end_idx = min(entry_idx + max_bars, len(df) - 1)
        forward = df.iloc[entry_idx + 1:end_idx + 1]

        dirs = [direction] if direction in {"long", "short"} else ["long", "short"]

        for dirn in dirs:
            if dirn == "long":
                sl_price = entry_price - sl_size
                tp_price = entry_price + tp_size
                hit_tp = hit_sl = False
                hit_time = None
                hit_price = None
                for idx2, row in forward.iterrows():
                    if row['high'] >= tp_price:
                        hit_tp = True
                        hit_time = idx2
                        hit_price = float(tp_price)
                        break
                    if row['low'] <= sl_price:
                        hit_sl = True
                        hit_time = idx2
                        hit_price = float(sl_price)
                        break
                if hit_tp and not hit_sl:
                    label = 1
                    end_time = hit_time
                    realized_return = (hit_price - entry_price) / sl_size
                elif hit_sl and not hit_tp:
                    label = 0
                    end_time = hit_time
                    realized_return = (hit_price - entry_price) / sl_size
                else:
                    label = 0
                    end_time = df.index[end_idx]
                    last_price = float(df['close'].iat[end_idx])
                    realized_return = (last_price - entry_price) / sl_size
            else:  # short
                sl_price = entry_price + sl_size
                tp_price = entry_price - tp_size
                hit_tp = hit_sl = False
                hit_time = None
                hit_price = None
                for idx2, row in forward.iterrows():
                    if row['low'] <= tp_price:
                        hit_tp = True
                        hit_time = idx2
                        hit_price = float(tp_price)
                        break
                    if row['high'] >= sl_price:
                        hit_sl = True
                        hit_time = idx2
                        hit_price = float(sl_price)
                        break
                if hit_tp and not hit_sl:
                    label = 1
                    end_time = hit_time
                    realized_return = (entry_price - hit_price) / sl_size
                elif hit_sl and not hit_tp:
                    label = 0
                    end_time = hit_time
                    realized_return = (entry_price - hit_price) / sl_size
                else:
                    label = 0
                    end_time = df.index[end_idx]
                    last_price = float(df['close'].iat[end_idx])
                    realized_return = (entry_price - last_price) / sl_size

            duration = (pd.to_datetime(end_time) - pd.to_datetime(t)).total_seconds() if end_time is not None else np.nan

            rec = {
                "candidate_time": t,
                "entry_price": entry_price,
                "atr": atr_t,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "end_time": end_time,
                "label": int(label),
                "duration": duration,
                "realized_return": float(realized_return),
                "direction": dirn,
                "tick_rate": np.nan,
                "uptick_ratio": np.nan,
                "buy_vol_ratio": np.nan,
                "micro_range": np.nan,
                "rvol_micro": float(df['rvol'].iat[entry_idx]) if 'rvol' in df.columns else np.nan,
            }
            records.append(rec)

    if not records:
        return pd.DataFrame(columns=[
            "candidate_time","entry_price","atr","sl_price","tp_price","end_time","label","duration","realized_return",
            "direction","tick_rate","uptick_ratio","buy_vol_ratio","micro_range","rvol_micro"
        ])

    cand_df = pd.DataFrame.from_records(records)
    cand_df['candidate_time'] = pd.to_datetime(cand_df['candidate_time'])
    cand_df = cand_df.set_index('candidate_time').sort_index()
    return cand_df