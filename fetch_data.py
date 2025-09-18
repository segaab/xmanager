# ───────────────────────────────────────── fetch_data.py ─────────────────────────────────────────
"""
Robust data-fetch helpers.

Changes vs. previous version
1. Gracefully handles Yahoo intraday limits (≤ 30 d for 1 m, ≤ 60 d for 5 m/15 m/30 m/60 m).
   • Long intraday date-ranges are split into smaller windows and concatenated.
2. Tries yahooquery first, transparently falls back to yfinance download.
3. Guarantees a canonical OHLCV index (UTC, tz-aware, no duplicated rows).
4. Added simple in-memory cache to avoid re-hitting the API inside the same Streamlit session.
"""

from __future__ import annotations

import functools
import os
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from sodapy import Socrata

# 1️⃣  market data back-ends ----------------------------------------------------
from yahooquery import Ticker
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
Intraday = Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
_DAY = timedelta(days=1)


@functools.lru_cache(maxsize=128)  # very small cache – enough for interactive use
def _fetch_chunk(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Fetch a SINGLE (small) date-range from Yahoo Finance.

    First yahooquery (fast), on failure yfinance (stable).
    """
    # ---------------- yahooquery ----------------
    try:
        tq = Ticker(symbol, asynchronous=False)
        df = tq.history(start=start, end=end, interval=interval)
        if isinstance(df, dict):          # happens for multi-symbol calls
            df = df.get(symbol, pd.DataFrame())
        if not df.empty:
            return df.reset_index()
    except Exception:
        pass  # silently fall through to yfinance

    # ---------------- yfinance fallback ---------
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            return df
    except Exception as e:
        raise RuntimeError(f"Both yahooquery and yfinance failed: {e}")

    # final safety
    return pd.DataFrame()


def fetch_price(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV DataFrame indexed by UTC datetime.

    Parameters
    ----------
    symbol   : Yahoo symbol (e.g. 'GC=F')
    start    : 'YYYY-MM-DD'
    end      : 'YYYY-MM-DD'   (exclusive, like pandas/yahoo)
    interval : Yahoo interval string

    Returns
    -------
    DataFrame with columns ['open','high','low','close','volume'] and tz-aware index.
    """

    # ------------------------------------------------------------------ helpers
    def _needs_chunking(ivl: str) -> bool:
        return ivl not in {"1d", "1wk", "1mo"}

    def _max_span(ivl: str) -> timedelta:
        # Yahoo hard limits
        if ivl == "1m":
            return timedelta(days=30)
        if ivl in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
            return timedelta(days=60)
        return timedelta(days=3650)  # effectively unlimited for daily+

    # ------------------------------------------------------------------ split logic
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    if start_dt >= end_dt:
        raise ValueError("`start` must be < `end`")

    chunks: list[pd.DataFrame] = []
    span = _max_span(interval)

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
            chunks.append(df_chunk)
            ptr = chunk_end + _DAY  # move one day forward to avoid overlap
    else:
        chunks.append(_fetch_chunk(symbol, start, end, interval))

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        return df  # caller will handle

    # ------------------------------------------------------------------ canonicalise
    # tolerate different column names from the two libraries
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjclose",
        "Volume": "volume",
        "close": "close",
        "high": "high",
        "low": "low",
        "open": "open",
        "volume": "volume",
    }
    df.rename(columns=col_map, inplace=True)

    # prefer 'adjclose' if plain 'close' missing
    if "close" not in df.columns and "adjclose" in df.columns:
        df["close"] = df["adjclose"]

    keep = ["open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan

    # make index tz-aware UTC
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], utc=True)
    elif "Datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    elif "datetime" not in df.columns:
        raise RuntimeError("Could not find datetime column after download.")

    df = (
        df.set_index("datetime")
        .sort_index()
        .loc[:, keep]
        .dropna(how="all")  # drop rows that are all NaN (can happen during holidays)
    )

    # remove duplicates introduced at chunk boundaries
    df = df[~df.index.duplicated(keep="first")]

    return df


# ───────────────────────────────── COT helpers (unchanged API) ────────────────
def init_socrata_client():
    token = os.getenv("SOCRATA_APP_TOKEN")
    return Socrata("publicreporting.cftc.gov", token)


def fetch_cot(
    client: Socrata | None = None,
    start: str | None = None,
    end: str | None = None,
    cot_name: str | None = None,
) -> pd.DataFrame:
    """
    Identical public API – just a tiny bug-fix: *numeric_cols* must ignore the date column
    when to_numeric() fails.
    """
    client = client or init_socrata_client()

    where = []
    if start:
        where.append(f"report_date_as_yyyy_mm_dd >= '{start}'")
    if end:
        where.append(f"report_date_as_yyyy_mm_dd <= '{end}'")
    if cot_name:
        where.append(f"market_and_contract_description = '{cot_name}'")

    res = client.get("6dca-aqww", where=" AND ".join(where) if where else "", limit=50_000)
    df = pd.DataFrame.from_records(res)
    if df.empty:
        return df

    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    numeric_cols = [c for c in df.columns if c != "report_date_as_yyyy_mm_dd"]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = (
        df.groupby("report_date_as_yyyy_mm_dd")[numeric_cols]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"report_date_as_yyyy_mm_dd": "report_date"})
        .sort_values("report_date")
    )
    return out
