# ───────────────────────────────────────── fetch_data.py ─────────────────────────────────────────
"""
Market-data utilities that USE **yahooquery only** (no yfinance fallback).

Fixes / features
• Works around Yahoo’s intraday limits (≤30 d for 1 m, ≤60 d for the other intraday bars)
  by automatically slicing long date-ranges into legal windows and stitching the chunks.
• Light in-memory cache so repeated requests inside one session are instant.
• Always returns a canonical OHLCV frame (UTC, tz-aware, no duplicates).
• COT helper got a minor numeric-column bug-fix (API unchanged).
"""

from __future__ import annotations

import functools
import os
import time
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from sodapy import Socrata
from yahooquery import Ticker

# --------------------------------------------------------------------------- #
Intraday = Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
_DAY = timedelta(days=1)


# ╭──────────────────────────────────────────────────────────────────────────╮
# │                       low-level chunk downloader                        │
# ╰──────────────────────────────────────────────────────────────────────────╯
@functools.lru_cache(maxsize=128)
def _fetch_chunk(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Download ONE legally-sized chunk via yahooquery and return a dataframe with the
    raw columns as delivered by the library.  The small LRU cache speeds up
    repetitive interactive use inside Streamlit.
    """
    tk = Ticker(symbol, asynchronous=False)          # synchronous avoids event-loop issues
    try:
        hist = tk.history(start=start, end=end, interval=interval)
    except Exception as e:
        raise RuntimeError(f"yahooquery error for {symbol}: {e}")

    # Multi-symbol dict → keep our symbol only
    if isinstance(hist, dict):
        hist = hist.get(symbol, pd.DataFrame())

    if hist.empty:
        return pd.DataFrame()

    # MultiIndex (symbol, date) → drop first level
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index(level=0, drop=True).reset_index()

    return hist.reset_index(drop=True)


# ╭──────────────────────────────────────────────────────────────────────────╮
# │                           public fetch_price()                           │
# ╰──────────────────────────────────────────────────────────────────────────╯
def fetch_price(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    symbol   : e.g. 'GC=F'
    start    : 'YYYY-MM-DD'
    end      : 'YYYY-MM-DD'  (exclusive, like pandas)
    interval : Yahoo interval string

    Returns
    -------
    tz-aware (UTC) DataFrame with columns [open, high, low, close, volume]
    """
    # ------------- helpers ----------------------------------------------------
    def _needs_chunking(ivl: str) -> bool:
        return ivl not in {"1d", "1wk", "1mo"}

    def _max_span(ivl: str) -> timedelta:
        if ivl == "1m":
            return timedelta(days=30)
        if ivl in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
            return timedelta(days=60)
        return timedelta(days=3650)  # ≈unlimited for daily+

    # ------------- argument hygiene ------------------------------------------
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    if start_dt >= end_dt:
        raise ValueError("`start` must be before `end`")

    # ------------- download (with automatic windowing) -----------------------
    span = _max_span(interval)
    dfs: list[pd.DataFrame] = []
    if _needs_chunking(interval) and end_dt - start_dt > span:
        ptr = start_dt
        while ptr < end_dt:
            chunk_end = min(ptr + span, end_dt)
            df_chunk = _fetch_chunk(
                symbol,
                ptr.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                interval,
            )
            dfs.append(df_chunk)
            ptr = chunk_end + _DAY     # avoid overlap
            time.sleep(0.15)           # polite pause to respect rate limits
    else:
        dfs.append(_fetch_chunk(symbol, start, end, interval))

    df_raw = pd.concat(dfs, ignore_index=True)
    if df_raw.empty:
        return df_raw

    # ------------- canonicalise columns --------------------------------------
    rename = {
        "date": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjclose": "adjclose",
        "volume": "volume",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjclose",
        "Volume": "volume",
    }
    df_raw.rename(columns=rename, inplace=True)

    if "datetime" not in df_raw.columns:
        raise RuntimeError("Could not locate a datetime column in yahooquery output.")

    # prefer adjclose if regular close missing
    if "close" not in df_raw.columns and "adjclose" in df_raw.columns:
        df_raw["close"] = df_raw["adjclose"]

    keep = ["open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    out = (
        df_raw.set_index(pd.to_datetime(df_raw["datetime"], utc=True))
        .sort_index()
        .loc[:, keep]
        .dropna(how="all")
    )

    # dedupe in case chunk borders overlapped
    out = out[~out.index.duplicated(keep="first")]

    return out


# ───────────────────────────────── COT helpers (tiny bug-fix) ────────────────
def init_socrata_client():
    return Socrata("publicreporting.cftc.gov", os.getenv("SOCRATA_APP_TOKEN"))


def fetch_cot(
    client: Socrata | None = None,
    start: str | None = None,
    end: str | None = None,
    cot_name: str | None = None,
) -> pd.DataFrame:
    client = client or init_socrata_client()
    where = []
    if start:
        where.append(f"report_date_as_yyyy_mm_dd >= '{start}'")
    if end:
        where.append(f"report_date_as_yyyy_mm_dd <= '{end}'")
    if cot_name:
        where.append(f"market_and_contract_description = '{cot_name}'")

    results = client.get("6dca-aqww", where=" AND ".join(where) if where else "", limit=50_000)
    df = pd.DataFrame.from_records(results)
    if df.empty:
        return df

    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    num_cols = [c for c in df.columns if c != "report_date_as_yyyy_mm_dd"]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (
        df.groupby("report_date_as_yyyy_mm_dd")[num_cols]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"report_date_as_yyyy_mm_dd": "report_date"})
        .sort_values("report_date")
    )
    return agg
