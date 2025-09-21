# labeling.py
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def generate_candidates_and_labels(
    df: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Generate trade candidates with ATR-based SL/TP and triple-barrier-like labeling.
    Ensures candidate_time and end_time are timezone-aware.
    """
    if df is None or df.empty:
        logger.warning("Input bars dataframe is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    if df.index.duplicated().any():
        logger.warning("Duplicate timestamps in bars — keeping first occurrence.")
        df = df[~df.index.duplicated(keep="first")]

    required_cols = {"high", "low", "close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"bars dataframe missing required columns: {missing_cols}")

    df["tr"] = _true_range(df["high"], df["low"], df["close"])
    df["atr"] = df["tr"].rolling(window=atr_window, min_periods=1).mean().fillna(method="ffill").fillna(0.0)

    candidates: List[Dict[str, Any]] = []
    n = len(df)

    for i in range(lookback, n):
        t = df.index[i]
        entry_price = float(df["close"].iat[i])
        atr_t = float(df["atr"].iat[i])

        if atr_t <= 0:
            continue

        if direction == "long":
            sl_price = entry_price - k_sl * atr_t
            tp_price = entry_price + k_tp * atr_t
        else:
            sl_price = entry_price + k_sl * atr_t
            tp_price = entry_price - k_tp * atr_t

        end_idx = min(i + max_bars, n - 1)
        label = 0
        hit_idx = end_idx
        hit_price = float(df["close"].iat[end_idx])

        for j in range(i + 1, end_idx + 1):
            px_high = float(df["high"].iat[j])
            px_low = float(df["low"].iat[j])
            px_close = float(df["close"].iat[j])

            if direction == "long":
                if px_high >= tp_price:
                    label = 1
                    hit_idx = j
                    hit_price = tp_price
                    break
                if px_low <= sl_price:
                    label = 0
                    hit_idx = j
                    hit_price = sl_price
                    break
            else:
                if px_low <= tp_price:
                    label = 1
                    hit_idx = j
                    hit_price = tp_price
                    break
                if px_high >= sl_price:
                    label = 0
                    hit_idx = j
                    hit_price = sl_price
                    break

        end_time = df.index[hit_idx]

        realized_return = (hit_price - entry_price) / entry_price if direction == "long" else (entry_price - hit_price) / entry_price
        duration_min = (end_time - t).total_seconds() / 60.0

        rec = {
            "candidate_time": t,
            "entry_price": entry_price,
            "atr": atr_t,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "end_time": end_time,
            "label": int(label),
            "duration": float(duration_min),
            "realized_return": float(realized_return),
            "direction": direction,
        }

        candidates.append(rec)

    candidates_df = pd.DataFrame(candidates)
    logger.info("Generated %d candidates", len(candidates_df))
    return candidates_df