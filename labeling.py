# labeling.py  (replace your existing function with this version)
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

# … imports stay the same …

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
    # … unchanged pre-amble …

    # ───────────────────────────────── record building ─────────────────────────
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

                # --- micro-features (now numeric placeholders, not NaN) --------
                "tick_rate": 0.0,
                "uptick_ratio": 0.0,
                "buy_vol_ratio": 0.0,
                "micro_range": 0.0,
                "rvol_micro": float(df["rvol"].iat[entry_idx])
                if "rvol" in df.columns
                else 0.0,
            }
    # … remainder unchanged …
