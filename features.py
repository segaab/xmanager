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
    This version is robust to duplicate/unsorted indexes by normalizing and deduping.
    """
    # --- Prepare price frame: ensure datetime index, sorted and unique -------------
    price_df = daily_bars.copy()
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    if price_df.index.duplicated().any():
        price_df = price_df[~price_df.index.duplicated(keep="last")]

    # baseline 35%
    gauge = pd.Series(0.35, index=price_df.index, dtype=float)

    # ---------- COT block (40 %) ---------------------------------------------
    if cot_df is not None and not cot_df.empty:
        cot_df = cot_df.copy()

        # if report_date column exists, convert and set as index; else ensure index is datetime
        if "report_date" in cot_df.columns:
            cot_df["report_date"] = pd.to_datetime(cot_df["report_date"])
            cot_df = cot_df.sort_values("report_date").set_index("report_date")
        else:
            if not isinstance(cot_df.index, pd.DatetimeIndex):
                cot_df.index = pd.to_datetime(cot_df.index)
            cot_df = cot_df.sort_index()

        # remove duplicate dates in cot_df index (keep last)
        if cot_df.index.duplicated().any():
            cot_df = cot_df[~cot_df.index.duplicated(keep="last")]

        # fetch long/short legs (multiple dataset aliases supported)
        nc_long = _safe_series_any(cot_df, ["noncommercial_long", "noncomm_positions_long_all"])
        nc_short = _safe_series_any(cot_df, ["noncommercial_short", "noncomm_positions_short_all"])
        c_long = _safe_series_any(cot_df, ["commercial_long", "comm_positions_long_all"])
        c_short = _safe_series_any(cot_df, ["commercial_short", "comm_positions_short_all"])
        # open_interest optionally used elsewhere
        open_int = _safe_series_any(cot_df, ["open_interest", "open_interest_all"], default=np.nan)

        # net & net-change (index == cot_df.index)
        nc_net_chg = (nc_long - nc_short).diff().fillna(0)
        c_net_chg = (c_long - c_short).diff().fillna(0)

        # ensure unique indexes on the net-change (should be unique after deduping cot_df)
        if nc_net_chg.index.duplicated().any():
            nc_net_chg = nc_net_chg[~nc_net_chg.index.duplicated(keep="last")]
        if c_net_chg.index.duplicated().any():
            c_net_chg = c_net_chg[~c_net_chg.index.duplicated(keep="last")]

        # normalise to ±1 using MAD-scale for robustness
        def _norm(s: pd.Series) -> pd.Series:
            if s.empty:
                return pd.Series([], index=s.index, dtype=float)
            mad = s.abs().median()
            if mad == 0 or np.isnan(mad):
                return pd.Series(0, index=s.index, dtype=float)
            return (s / (5 * mad)).clip(-1, 1)  # 5×MAD ≈ 3 σ

        nc_norm = _norm(nc_net_chg)
        c_norm = _norm(c_net_chg)

        # guard against duplicated index on normalized series
        if nc_norm.index.duplicated().any():
            nc_norm = nc_norm[~nc_norm.index.duplicated(keep="last")]
        if c_norm.index.duplicated().any():
            c_norm = c_norm[~c_norm.index.duplicated(keep="last")]

        # align with daily price index and interpolate in time (requires datetime index)
        # reindex source (nc_norm) to target index (price_df.index)
        nc_interp = nc_norm.reindex(price_df.index).interpolate(method="time").fillna(0)
        c_interp = c_norm.reindex(price_df.index).interpolate(method="time").fillna(0)

        gauge += 0.20 * nc_interp
        gauge += 0.20 * c_interp

    else:
        # no cot data: zero series for net-change with price index
        nc_net_chg = pd.Series(0, index=price_df.index, dtype=float)
        c_net_chg = pd.Series(0, index=price_df.index, dtype=float)

    # ---------- Momentum / realised-vol block (25 %) --------------------------
    # make sure 'close' exists
    if "close" not in price_df.columns:
        raise KeyError("daily_bars must contain a 'close' column")

    ret = price_df["close"].pct_change().fillna(0)
    mom = ret.rolling(mom_window, min_periods=1).mean()
    vol = ret.rolling(mom_window, min_periods=1).std()

    mom_norm = (mom / (mom.abs().max() or 1)).clip(-1, 1)
    vol_norm = (vol / (vol.max() or 1)).clip(0, 1)

    gauge += 0.15 * mom_norm
    gauge -= 0.10 * vol_norm

    # ---------- Final clip & return ------------------------------------------
    # reindex raw net-change series to the price index (safe because we ensured uniqueness above)
    try:
        nc_out = nc_net_chg.reindex(price_df.index).fillna(0)
        c_out = c_net_chg.reindex(price_df.index).fillna(0)
    except ValueError:
        # last-resort: if something unexpected still has dupes, produce zero-series
        nc_out = pd.Series(0, index=price_df.index, dtype=float)
        c_out = pd.Series(0, index=price_df.index, dtype=float)

    out = pd.DataFrame(
        {
            "health_gauge": gauge.clip(0, 1),
            "noncomm_net_chg": nc_out,
            "comm_net_chg": c_out,
        },
        index=price_df.index,
    )
    return out