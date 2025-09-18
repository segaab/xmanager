# labeling.py
"""
Candidate generation and triple-barrier labeling (updated).

Behavior (key points)
- SL = latest ATR(window=atr_window) at candidate time (same interval as bars).
- TP = 3 * SL (i.e. target = entry ± 3 * ATR).
- Labels:
    1 -> TP hit before SL within the horizon (max_bars)
    0 -> SL hit first OR neither barrier hit within horizon (vertical hit treated as loss/neutral)
- Produces a candidates DataFrame with columns:
    ['candidate_time','entry_price','atr','sl_price','tp_price','end_time','label','duration','realized_return']
  plus some lightweight microfeature placeholders (tick_rate, uptick_ratio, buy_vol_ratio,
  micro_range, rvol_micro) so downstream code has expected columns. Fill-ins are NaN if unavailable.

Notes
- This function is deliberately conservative: it skips candidates if ATR is NaN or not positive.
- It assumes `bars` is a tz-aware pandas DataFrame indexed by datetime with columns:
  ['open','high','low','close','volume'] (and optional 'rvol' if available).
- Intended to be compatible with the existing Streamlit app pipeline.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np


def _compute_atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute ATR (simple rolling average of True Range) aligned with bars index.
    """
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
    atr_window: int = 14,
    max_bars: int = 60,
    rvol_threshold: float = 1.5,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Generate candidate entries and label them using SL = ATR(atr_window) and TP = 3*SL.

    Parameters
    ----------
    bars : DataFrame
        OHLCV bars indexed by tz-aware datetime, columns: open, high, low, close, volume.
        May include 'rvol' column (relative volume). If not present, rvol is computed from volume.
    lookback : int
        Minimum number of bars of history before considering candidate (used for some heuristics).
    atr_window : int
        ATR window in bars (e.g., 14).
    max_bars : int
        Vertical barrier in bars (maximum holding time).
    rvol_threshold : float
        Simple heuristic: only create candidates where rvol >= rvol_threshold.
    direction : {"long","short","both"}
        Which trade directions to consider. For "both" we will create two candidates per time (with
        different TP/SL). Default is "long".

    Returns
    -------
    DataFrame of candidates (see docstring for columns).
    """
    # Basic sanity checks
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must be indexed by a pandas.DatetimeIndex")

    # Ensure required columns exist
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(set(bars.columns)):
        raise ValueError(f"`bars` must contain columns: {required}")

    df = bars.copy().sort_index()

    # compute rvol if missing: volume / rolling_mean(volume,20)
    if 'rvol' not in df.columns:
        vol_mean = df['volume'].rolling(window=20, min_periods=1).mean()
        df['rvol'] = df['volume'] / (vol_mean.replace(0, np.nan))

    # compute ATR
    atr = _compute_atr(df, window=atr_window)
    df['atr'] = atr

    # candidates heuristic: rvol >= threshold AND not in the first `lookback` bars
    cand_mask = (df['rvol'] >= rvol_threshold) & (np.arange(len(df)) >= lookback)
    candidate_times = df.index[cand_mask]

    records = []
    for t in candidate_times:
        entry_idx = df.index.get_loc(t)
        entry_price = float(df['close'].iat[entry_idx])  # use close at t as entry price
        atr_t = float(df['atr'].iat[entry_idx]) if not pd.isna(df['atr'].iat[entry_idx]) else np.nan

        # skip if ATR invalid
        if not np.isfinite(atr_t) or atr_t <= 0:
            continue

        # horizon end index (exclusive)
        end_idx = min(entry_idx + max_bars, len(df) - 1)

        # slice forward bars for simulation (from next bar up to end_idx)
        forward = df.iloc[entry_idx + 1:end_idx + 1]  # may be empty if at tail

        # if both directions requested, produce two candidates (long and short)
        dirs = [direction] if direction in {"long", "short"} else ["long", "short"]

        for dirn in dirs:
            if dirn == "long":
                sl_price = entry_price - atr_t
                tp_price = entry_price + 3.0 * atr_t
                hit_tp = False
                hit_sl = False
                hit_time = None
                hit_price = None
                # simulate forward: check first time high >= tp or low <= sl
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
                # decide label
                if hit_tp and not hit_sl:
                    label = 1
                    end_time = hit_time
                    realized_return = (hit_price - entry_price) / atr_t  # in R units
                elif hit_sl and not hit_tp:
                    label = 0
                    end_time = hit_time
                    realized_return = (hit_price - entry_price) / atr_t
                else:
                    # no barrier reached within horizon -> treat as loss/neutral (label 0)
                    label = 0
                    end_time = df.index[end_idx]
                    # realized return relative to ATR from last available price
                    last_price = float(df['close'].iat[end_idx])
                    realized_return = (last_price - entry_price) / atr_t
            else:  # short
                sl_price = entry_price + atr_t
                tp_price = entry_price - 3.0 * atr_t
                hit_tp = False
                hit_sl = False
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
                    realized_return = (entry_price - hit_price) / atr_t  # positive if win
                elif hit_sl and not hit_tp:
                    label = 0
                    end_time = hit_time
                    realized_return = (entry_price - hit_price) / atr_t
                else:
                    label = 0
                    end_time = df.index[end_idx]
                    last_price = float(df['close'].iat[end_idx])
                    realized_return = (entry_price - last_price) / atr_t

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
                # lightweight microfeature placeholders (downstream expects these names)
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
    # keep dtype tidy
    cand_df['candidate_time'] = pd.to_datetime(cand_df['candidate_time'])
    cand_df = cand_df.set_index('candidate_time').sort_index()

    return cand_df