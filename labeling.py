# labeling.py
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

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
    rvol_threshold: float = 1.5,
    direction: str = "long",
) -> pd.DataFrame:
    """
    Generate trade candidates with ATR-based SL/TP and triple-barrier-like labeling using a fixed-horizon.

    - SL = entry_price +/- k_sl * ATR(atr_window)
    - TP = entry_price +/- k_tp * ATR(atr_window)  (k_tp typically = 3.0 for 3R)
    - Label = 1 if TP is hit before SL within max_bars forward bars, else 0.
    - If neither barrier is hit before max_bars, label = 0 (vertical barrier).
    - Micro-features are filled with numeric defaults (0.0) if missing.

    Returns DataFrame with candidate records (one row per candidate).
    """
    if df is None or df.empty:
        logger.warning("Input bars dataframe is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # ensure index is datetime and sorted
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    if df.index.duplicated().any():
        logger.warning("Duplicate timestamps in bars — keeping first occurrence.")
        df = df[~df.index.duplicated(keep="first")]

    required_cols = {"high", "low", "close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"bars dataframe missing required columns for ATR calculation: {missing_cols}")

    # True Range -> ATR (simple rolling mean of TR)
    df["tr"] = _true_range(df["high"], df["low"], df["close"])
    df["atr"] = df["tr"].rolling(window=atr_window, min_periods=1).mean().fillna(method="ffill").fillna(0.0)

    candidates: List[Dict[str, Any]] = []

    n = len(df)
    # start from index position `lookback` to ensure prior context if needed
    for i in range(lookback, n):
        t = df.index[i]
        entry_price = float(df["close"].iat[i])
        atr_t = float(df["atr"].iat[i]) if not pd.isna(df["atr"].iat[i]) else 0.0

        # skip if ATR is zero or non-positive (data problem)
        if atr_t <= 0:
            continue

        # compute SL/TP based on direction
        if direction == "long":
            sl_price = entry_price - k_sl * atr_t
            tp_price = entry_price + k_tp * atr_t
        else:
            sl_price = entry_price + k_sl * atr_t
            tp_price = entry_price - k_tp * atr_t

        # search forward up to max_bars to find first barrier hit
        end_idx = min(i + max_bars, n - 1)
        label = 0
        hit_idx = end_idx
        hit_price = float(df["close"].iat[end_idx])

        for j in range(i + 1, end_idx + 1):
            px_high = float(df["high"].iat[j])
            px_low = float(df["low"].iat[j])
            px_close = float(df["close"].iat[j])

            if direction == "long":
                # if high >= tp_price -> TP hit
                if px_high >= tp_price:
                    label = 1
                    hit_idx = j
                    # approximate hit price as min(high, tp_price) -> use tp_price for return calc
                    hit_price = tp_price
                    break
                # if low <= sl_price -> SL hit
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
        # realized return based on hit_price (or close at vertical if no hit)
        realized_return = (hit_price - entry_price) / entry_price if direction == "long" else (entry_price - hit_price) / entry_price
        duration_min = (end_time - t).total_seconds() / 60.0

        # micro-features: prefer existing columns, else default 0.0
        tick_rate = float(df["tick_rate"].iat[i]) if "tick_rate" in df.columns else 0.0
        uptick_ratio = float(df["uptick_ratio"].iat[i]) if "uptick_ratio" in df.columns else 0.0
        buy_vol_ratio = float(df["buy_vol_ratio"].iat[i]) if "buy_vol_ratio" in df.columns else 0.0
        micro_range = float(df["micro_range"].iat[i]) if "micro_range" in df.columns else 0.0
        rvol_micro = float(df["rvol"].iat[i]) if "rvol" in df.columns else 0.0

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
            # micro-features (numeric, no NaN)
            "tick_rate": tick_rate,
            "uptick_ratio": uptick_ratio,
            "buy_vol_ratio": buy_vol_ratio,
            "micro_range": micro_range,
            "rvol_micro": rvol_micro,
        }

        candidates.append(rec)

    candidates_df = pd.DataFrame(candidates)

    # Diagnostics logging
    logger.info("Candidate labeling diagnostics")
    logger.info("Total candidates produced: %d", len(candidates_df))
    if not candidates_df.empty:
        logger.info("Label value counts: %s", candidates_df["label"].value_counts(dropna=False).to_dict())
        nan_counts = {c: int(candidates_df[c].isna().sum()) for c in ["entry_price", "atr", "sl_price", "tp_price"] if c in candidates_df.columns}
        logger.info("NaN counts for important fields: %s", nan_counts)

    return candidates_df