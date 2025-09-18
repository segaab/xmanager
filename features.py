# ─────────────────────────────────────────── features.py ─────────────────────────────────────────
"""
Light feature improvements
1. compute_rvol()  :  volume_avg now uses *median* when data is spiky, and we avoid
                      divide-by-zero with a tiny epsilon.
2. calculate_health_gauge() gets an additional 25 % weight from recent momentum and
   volatility – giving the gauge some life even when COT is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    # median is often more robust for futures volume
    out["volume_avg"] = out["volume"].rolling(window, min_periods=1).median()
    eps = 1e-9
    out["rvol"] = out["volume"] / (out["volume_avg"] + eps)
    return out


def calculate_health_gauge(
    cot_df: pd.DataFrame,
    daily_bars: pd.DataFrame,
    mom_window: int = 14,
) -> pd.DataFrame:
    df = daily_bars.copy()

    # ---------- base gauge ----------------------------------------------------
    gauge = pd.Series(0.5, index=df.index, dtype=float)

    # COT contribution (50 %)
    if not cot_df.empty:
        cot_df = cot_df.sort_values("report_date")
        net = (
            cot_df.get("commercial_long", 0) - cot_df.get("commercial_short", 0)
        ) / (cot_df.get("commercial_long", 1) + cot_df.get("commercial_short", 1))
        net = net.clip(-1, 1)
        cot_interp = (
            net.rename("cot")
            .reindex(df.index)
            .interpolate(method="time")
            .fillna(0)
        )
        gauge += 0.5 * cot_interp

    # Price momentum & realised volatility (25 %)
    ret = df["close"].pct_change().fillna(0)
    mom = ret.rolling(mom_window).mean()
    vol = ret.rolling(mom_window).std()

    # normalise to [-1,1] then to [0,1]
    mom_norm = (mom / mom.abs().max()).clip(-1, 1).fillna(0)
    vol_norm = (vol / vol.abs().max()).clip(0, 1).fillna(0)

    gauge += 0.15 * mom_norm  # positive momentum = healthier
    gauge -= 0.10 * vol_norm  # very high vol = stress

    gauge = gauge.clip(0, 1)
    return pd.DataFrame({"health_gauge": gauge})
