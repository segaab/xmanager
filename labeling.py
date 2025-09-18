# labeling.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
    Generate trade candidates with entry, stop, target, and labels.
    Micro-features are numeric and default to 0.0 if missing.
    """
    candidates = []

    if df.empty:
        logging.warning("Input bars dataframe is empty. Returning empty candidates.")
        return pd.DataFrame()

    # Example rolling ATR calculation
    df["atr"] = df["close"].rolling(atr_window).apply(
        lambda x: x.max() - x.min(), raw=False
    )

    # Candidate generation loop (simplified)
    for i in range(lookback, len(df)):
        t = df.index[i]
        entry_idx = i
        entry_price = df["close"].iat[entry_idx]
        atr_t = df["atr"].iat[entry_idx]
        sl_price = entry_price - k_sl * atr_t if direction == "long" else entry_price + k_sl * atr_t
        tp_price = entry_price + k_tp * atr_t if direction == "long" else entry_price - k_tp * atr_t
        end_idx = min(i + max_bars, len(df) - 1)
        end_time = df.index[end_idx]
        realized_return = (df["close"].iat[end_idx] - entry_price) / entry_price
        label = int(realized_return >= (tp_price - entry_price) / entry_price)
        duration = (end_time - t).total_seconds() / 60
        dirn = direction

        # ───────────────────────────────── record building ─────────────────────────
        rec = {
            "candidate_time": t,
            "entry_price": float(entry_price),
            "atr": float(atr_t),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "end_time": end_time,
            "label": int(label),
            "duration": float(duration),
            "realized_return": float(realized_return),
            "direction": dirn,

            # --- micro-features (numeric, no NaN) -------------------------------
            "tick_rate": float(df["tick_rate"].iat[entry_idx]) if "tick_rate" in df.columns else 0.0,
            "uptick_ratio": float(df["uptick_ratio"].iat[entry_idx]) if "uptick_ratio" in df.columns else 0.0,
            "buy_vol_ratio": float(df["buy_vol_ratio"].iat[entry_idx]) if "buy_vol_ratio" in df.columns else 0.0,
            "micro_range": float(df["micro_range"].iat[entry_idx]) if "micro_range" in df.columns else 0.0,
            "rvol_micro": float(df["rvol"].iat[entry_idx]) if "rvol" in df.columns else 0.0,
        }

        candidates.append(rec)

    candidates_df = pd.DataFrame(candidates)

    # Logging diagnostics
    logging.info("Candidate labeling diagnostics")
    logging.info(f"Total candidates produced: {len(candidates_df)}")
    logging.info(f"Candidate columns:\n{list(candidates_df.columns)}")
    if "label" in candidates_df.columns:
        logging.info(f"label dtype: {candidates_df['label'].dtype}")
        logging.info(f"label unique values (sample):\n{candidates_df['label'].unique()[:10]}")
        logging.info(f"label value counts (including NaN):\n{candidates_df['label'].value_counts(dropna=False)}")
        logging.info(f"label null count: {candidates_df['label'].isna().sum()}")
    # Check NaNs for key fields
    key_fields = ["entry_price","atr","sl_price","tp_price"]
    nan_counts = {k: candidates_df[k].isna().sum() for k in key_fields if k in candidates_df.columns}
    logging.info(f"NaN counts for important fields:\n{nan_counts}")

    return candidates_df