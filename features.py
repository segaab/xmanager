"""
Feature helpers – *HealthGauge v2*

Adds:
• _safe_series_any()  – choose the first existing column name.
• non-commercial / commercial **net-change** features.
• HealthGauge now blends:
    – non-comm net-change   (20 %)
    – comm     net-change   (20 %)
    – price momentum        (15 %)
    – realised volatility  −(10 %)
    – neutral baseline      (35 %)
  and returns the two net-change series so they can be joined to the
  intraday candidate frame used by the entry model.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------#
def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return df[col] coerced to numeric; if missing → Series[default]."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)

def _safe_series_any(df: pd.DataFrame, cols: list[str], default: float = 0.0) -> pd.Series:
    """Pick the first existing column in *cols*; else return default series."""
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)

# -----------------------------------------------------------------------------#
def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["volume_avg"] = out["volume"].rolling(window, min_periods=1).median()
    out["rvol"] = out["volume"] / (out["volume_avg"] + 1e-9)
    return out

# -----------------------------------------------------------------------------#
def calculate_health_gauge(
    cot_df: pd.DataFrame,
    daily_bars: pd.DataFrame,
    mom_window: int = 14,
) -> pd.DataFrame:
    """
    Creates a daily dataframe with
        • health_gauge           ∊ [0,1]
        • noncomm_net_chg        (raw # contracts)
        • comm_net_chg           (raw # contracts)
    """
    price_df = daily_bars.copy()
    gauge = pd.Series(0.35, index=price_df.index, dtype=float)  # baseline 35 %

    # ---------- COT block (40 %) ---------------------------------------------
    if not cot_df.empty:
        cot_df = cot_df.sort_values("report_date").set_index("report_date")

        # fetch long/short legs (multiple dataset aliases supported)
        nc_long   = _safe_series_any(cot_df, ["noncommercial_long", "noncomm_positions_long_all"])
        nc_short  = _safe_series_any(cot_df, ["noncommercial_short", "noncomm_positions_short_all"])
        c_long    = _safe_series_any(cot_df, ["commercial_long", "comm_positions_long_all"])
        c_short   = _safe_series_any(cot_df, ["commercial_short", "comm_positions_short_all"])
        open_int  = _safe_series_any(cot_df, ["open_interest", "open_interest_all"], default=np.nan)

        # net & **net-change**
        nc_net_chg = (nc_long - nc_short).diff().fillna(0)
        c_net_chg  = (c_long - c_short).diff().fillna(0)

        # normalise to ±1 using MAD-scale for robustness
        def _norm(s: pd.Series) -> pd.Series:
            mad = s.abs().median()
            if mad == 0:
                return pd.Series(0, index=s.index)
            return (s / (5 * mad)).clip(-1, 1)          # 5×MAD ≈ 3 σ

        nc_norm = _norm(nc_net_chg)
        c_norm  = _norm(c_net_chg)

        # align with daily price index
        nc_interp = (
            nc_norm.rename("nc_norm")
            .reindex(price_df.index)
            .interpolate("time")
            .fillna(0)
        )
        c_interp = (
            c_norm.rename("c_norm")
            .reindex(price_df.index)
            .interpolate("time")
            .fillna(0)
        )

        gauge += 0.20 * nc_interp
        gauge += 0.20 * c_interp

    else:
        nc_net_chg = pd.Series(0, index=price_df.index)
        c_net_chg  = pd.Series(0, index=price_df.index)

    # ---------- Momentum / realised-vol block (25 %) --------------------------
    ret = price_df["close"].pct_change().fillna(0)
    mom = ret.rolling(mom_window).mean()
    vol = ret.rolling(mom_window).std()

    mom_norm = (mom / (mom.abs().max() or 1)).clip(-1, 1)
    vol_norm = (vol / (vol.max() or 1)).clip(0, 1)

    gauge += 0.15 * mom_norm
    gauge -= 0.10 * vol_norm

    # ---------- Final clip & return ------------------------------------------
    out = pd.DataFrame(
        {
            "health_gauge": gauge.clip(0, 1),
            "noncomm_net_chg": nc_net_chg.reindex(price_df.index).fillna(0),
            "comm_net_chg":    c_net_chg.reindex(price_df.index).fillna(0),
        }
    )
    return out
