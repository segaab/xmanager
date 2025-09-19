# features.py
from __future__ import annotations
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return df[col] coerced to numeric; if missing → Series[default]."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _safe_series_any(df: pd.DataFrame, cols: List[str], default: float = 0.0) -> pd.Series:
    """Pick the first existing column in *cols*; else return default series."""
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute relative volume (rvol) as volume / rolling_median(volume, window).
    Adds 'volume_avg' and 'rvol' to a copy of df.
    """
    out = df.copy()
    if "volume" not in out.columns:
        logger.warning("compute_rvol: 'volume' column missing. Adding zeros.")
        out["volume"] = 0.0
    out["volume_avg"] = out["volume"].rolling(window, min_periods=1).median()
    out["rvol"] = out["volume"] / (out["volume_avg"] + 1e-9)
    out["rvol"] = out["rvol"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def calculate_health_gauge(
    cot_df: pd.DataFrame,
    daily_bars: pd.DataFrame,
    mom_window: int = 14,
) -> pd.DataFrame:
    """
    Creates a daily dataframe with:
        - health_gauge in [0,1]
        - noncomm_net_chg (raw)
        - comm_net_chg (raw)

    Robust to duplicate/unsorted indexes by normalizing and deduping.
    """
    if daily_bars is None or daily_bars.empty:
        raise ValueError("daily_bars must be a non-empty DataFrame with a datetime index and 'close' column.")

    price_df = daily_bars.copy()
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    if price_df.index.duplicated().any():
        logger.warning("calculate_health_gauge: duplicate indices in daily_bars — keeping last occurrence.")
        price_df = price_df[~price_df.index.duplicated(keep="last")]

    # baseline
    gauge = pd.Series(0.35, index=price_df.index, dtype=float)

    # COT block (40%)
    if cot_df is not None and not cot_df.empty:
        cot = cot_df.copy()
        if "report_date" in cot.columns:
            cot["report_date"] = pd.to_datetime(cot["report_date"])
            cot = cot.sort_values("report_date").set_index("report_date")
        else:
            if not isinstance(cot.index, pd.DatetimeIndex):
                cot.index = pd.to_datetime(cot.index)
            cot = cot.sort_index()

        if cot.index.duplicated().any():
            logger.warning("calculate_health_gauge: duplicate indices in COT — keeping last.")
            cot = cot[~cot.index.duplicated(keep="last")]

        nc_long = _safe_series_any(cot, ["noncommercial_long", "noncomm_positions_long_all", "noncommercial_long_all"])
        nc_short = _safe_series_any(cot, ["noncommercial_short", "noncomm_positions_short_all", "noncommercial_short_all"])
        c_long = _safe_series_any(cot, ["commercial_long", "comm_positions_long_all", "commercial_long_all"])
        c_short = _safe_series_any(cot, ["commercial_short", "comm_positions_short_all", "commercial_short_all"])

        nc_net = (nc_long - nc_short).fillna(0.0)
        c_net = (c_long - c_short).fillna(0.0)

        # net change
        nc_net_chg = nc_net.diff().fillna(0.0)
        c_net_chg = c_net.diff().fillna(0.0)

        def _norm(s: pd.Series) -> pd.Series:
            if s.empty:
                return pd.Series(0.0, index=price_df.index, dtype=float)
            mad = s.abs().median()
            if mad == 0 or np.isnan(mad):
                return pd.Series(0.0, index=s.index, dtype=float)
            return (s / (5.0 * mad)).clip(-1.0, 1.0)

        nc_norm = _norm(nc_net_chg)
        c_norm = _norm(c_net_chg)

        try:
            nc_interp = nc_norm.reindex(price_df.index).interpolate(method="time").fillna(0.0)
            c_interp = c_norm.reindex(price_df.index).interpolate(method="time").fillna(0.0)
        except Exception:
            nc_interp = nc_norm.reindex(price_df.index).ffill().fillna(0.0)
            c_interp = c_norm.reindex(price_df.index).ffill().fillna(0.0)

        gauge += 0.20 * nc_interp
        gauge += 0.20 * c_interp

        nc_out = nc_net_chg.reindex(price_df.index).fillna(0.0)
        c_out = c_net_chg.reindex(price_df.index).fillna(0.0)
    else:
        nc_out = pd.Series(0.0, index=price_df.index, dtype=float)
        c_out = pd.Series(0.0, index=price_df.index, dtype=float)

    # momentum / vol block (25%)
    if "close" not in price_df.columns:
        raise KeyError("daily_bars must contain 'close' for health gauge computation.")

    ret = price_df["close"].pct_change().fillna(0.0)
    mom = ret.rolling(mom_window, min_periods=1).mean()
    vol = ret.rolling(mom_window, min_periods=1).std()

    mom_norm = (mom / (mom.abs().max() or 1.0)).clip(-1.0, 1.0)
    vol_norm = (vol / (vol.max() or 1.0)).clip(0.0, 1.0)

    gauge += 0.15 * mom_norm
    gauge -= 0.10 * vol_norm

    gauge = gauge.clip(0.0, 1.0)

    out = pd.DataFrame(
        {
            "health_gauge": gauge.astype(float),
            "noncomm_net_chg": nc_out.astype(float),
            "comm_net_chg": c_out.astype(float),
        },
        index=price_df.index,
    )

    return out