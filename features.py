# ─────────────────────────────────────────── features.py ─────────────────────────────────────────
"""
Feature helpers (robust version)

Fixes
• calculate_health_gauge(): handles missing COT columns gracefully and never
  produces scalar objects that break .clip().
• compute_rvol(): same as before (uses rolling median, epsilon to avoid ÷0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------#
# Utility: always return a Series with the right index
def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """
    Return df[col] coerced to numeric; if column absent return a Series of
    `default` with the same index length.  Guarantees a pandas Series output.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


# -----------------------------------------------------------------------------#
def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["volume_avg"] = out["volume"].rolling(window, min_periods=1).median()
    out["rvol"] = out["volume"] / (out["volume_avg"] + 1e-9)
    return out


def calculate_health_gauge(
    cot_df: pd.DataFrame,
    daily_bars: pd.DataFrame,
    mom_window: int = 14,
) -> pd.DataFrame:
    """
    Combine COT information (if available) with simple price momentum/volatility
    proxies to form a [0,1] HealthGauge.
    """
    df = daily_bars.copy()
    gauge = pd.Series(0.5, index=df.index, dtype=float)  # neutral baseline

    # ------------------------------------------------------------------ COT (50 %)
    if not cot_df.empty:
        # ensure an index so _safe_series can mirror it
        cot_df = cot_df.sort_values("report_date").set_index("report_date")
        long = _safe_series(cot_df, "commercial_long")
        short = _safe_series(cot_df, "commercial_short")

        denom = (long + short).replace(0, np.nan)  # avoid division by zero
        net = ((long - short) / denom).fillna(0).clip(-1, 1)

        cot_interp = (
            net
            .rename("cot")
            .reindex(df.index)        # align to daily bars
            .interpolate(method="time")
            .fillna(0)
        )
        gauge += 0.5 * cot_interp

    # ---------------------------------------------------------------- momentum/vol (25 %)
    ret = df["close"].pct_change().fillna(0)
    mom = ret.rolling(mom_window).mean()
    vol = ret.rolling(mom_window).std()

    if mom.abs().max() != 0:
        mom_norm = (mom / mom.abs().max()).clip(-1, 1)
    else:
        mom_norm = pd.Series(0, index=df.index)

    if vol.max() != 0:
        vol_norm = (vol / vol.max()).clip(0, 1)
    else:
        vol_norm = pd.Series(0, index=df.index)

    gauge += 0.15 * mom_norm          # positive momentum adds health
    gauge -= 0.10 * vol_norm          # excessive realised vol subtracts health

    # ------------------------------------------------------------------ final clip
    return pd.DataFrame({"health_gauge": gauge.clip(0, 1)})
