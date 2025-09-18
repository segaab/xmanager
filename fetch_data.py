# fetch_data.py  (only the file contents shown — replace your existing file)
"""
Market-data utilities with threaded, batched fetching to support up to 15 years.

COT fetching now includes robust retry/backoff to reduce failures (ConnectionResetError, timeouts).
"""

from __future__ import annotations
import functools
import os
import time
from datetime import datetime, timedelta
from typing import Literal, List
import pandas as pd
import numpy as np
from sodapy import Socrata
from yahooquery import Ticker
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import socket

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --------------------------------------------------------------------------- #
Intraday = Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
_DAY = timedelta(days=1)


# ╭──────────────────────────────────────────────────────────────────────────╮
# │                       low-level chunk downloader                        │
# ╰──────────────────────────────────────────────────────────────────────────╯
@functools.lru_cache(maxsize=128)
def _fetch_chunk(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    tk = Ticker(symbol, asynchronous=False)
    try:
        hist = tk.history(start=start, end=end, interval=interval)
    except Exception as e:
        logging.error(f"Yahooquery error for {symbol} ({interval}): {e}")
        return pd.DataFrame()

    if isinstance(hist, dict):
        hist = hist.get(symbol, pd.DataFrame())

    if hist.empty:
        return pd.DataFrame()

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
    max_years: int = 15,
) -> pd.DataFrame:
    def _needs_chunking(ivl: str) -> bool:
        return ivl not in {"1d", "1wk", "1mo"}

    def _max_span(ivl: str) -> timedelta:
        if ivl == "1m":
            return timedelta(days=30)
        if ivl in {"2m","5m","15m","30m","60m","90m","1h"}:
            return timedelta(days=60)
        return timedelta(days=365)  # daily+: 1 year chunking

    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    if start_dt >= end_dt:
        raise ValueError("`start` must be before `end`")
    if (end_dt - start_dt).days > max_years * 366:
        logging.warning(f"Requested range > {max_years} years. Truncating to last {max_years} years.")
        start_dt = end_dt - timedelta(days=max_years*366)

    span = _max_span(interval)
    dfs: list[pd.DataFrame] = []

    if _needs_chunking(interval) and end_dt - start_dt > span:
        ptr = start_dt
        while ptr < end_dt:
            chunk_end = min(ptr + span, end_dt)
            df_chunk = _fetch_chunk(symbol, ptr.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), interval)
            dfs.append(df_chunk)
            ptr = chunk_end + _DAY
            time.sleep(0.15)
    else:
        dfs.append(_fetch_chunk(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), interval))

    df_raw = pd.concat(dfs, ignore_index=True)
    if df_raw.empty:
        return df_raw

    rename = {
        "date":"datetime","open":"open","high":"high","low":"low","close":"close",
        "adjclose":"adjclose","volume":"volume","Open":"open","High":"high","Low":"low",
        "Close":"close","Adj Close":"adjclose","Volume":"volume"
    }
    df_raw.rename(columns=rename, inplace=True)
    if "datetime" not in df_raw.columns:
        raise RuntimeError("Could not locate a datetime column in yahooquery output.")

    if "close" not in df_raw.columns and "adjclose" in df_raw.columns:
        df_raw["close"] = df_raw["adjclose"]

    keep = ["open","high","low","close","volume"]
    for col in keep:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    out = (
        df_raw.set_index(pd.to_datetime(df_raw["datetime"], utc=True))
        .sort_index()
        .loc[:, keep]
        .dropna(how="all")
    )

    out = out[~out.index.duplicated(keep="first")]
    return out


# ────────────────────────── Socrata helper with retry/backoff ───────────────
def init_socrata_client():
    """
    Initialize Socrata client. Use SOCRATA_APP_TOKEN in environment if available.
    """
    token = os.getenv("SOCRATA_APP_TOKEN")
    if token is None:
        logging.warning("Requests made without an app_token will be subject to strict throttling limits.")
    return Socrata("publicreporting.cftc.gov", token)


def _socrata_get_with_retry(client, dataset: str, where_clause: str, limit: int = 50_000,
                            retries: int = 4, backoff_base: float = 1.5, timeout: float = 30.0):
    """
    Wrapper around client.get with retries + exponential backoff.
    Catches common transient network errors including ConnectionResetError.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # Socrata's client.get does not accept timeout param directly in older versions;
            # rely on the client internals but catch network errors here.
            return client.get(dataset, where=where_clause, limit=limit)
        except Exception as exc:
            last_exc = exc
            # detect connection reset / transient network issues
            if isinstance(exc, (ConnectionResetError, socket.timeout, TimeoutError)):
                logging.warning(f"Socrata network error (attempt {attempt}/{retries}): {exc}")
            else:
                logging.warning(f"Socrata error (attempt {attempt}/{retries}): {exc}")
            sleep = backoff_base ** attempt
            logging.info(f"Retrying in {sleep:.1f}s...")
            time.sleep(sleep)
    logging.error(f"Socrata fetch failed after {retries} attempts: {last_exc}")
    raise last_exc


def fetch_cot(
    client: Socrata | None = None,
    start: str | None = None,
    end: str | None = None,
    cot_name: str | None = None,
    max_years: int = 15,
) -> pd.DataFrame:
    """
    Threaded, batched COT fetch for up to `max_years`. Uses _socrata_get_with_retry for robustness.
    """
    client = client or init_socrata_client()
    start_dt = pd.to_datetime(start) if start else pd.Timestamp.now() - timedelta(days=365 * max_years)
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.now()

    # truncate to max_years back from end_dt
    if (end_dt - start_dt).days > max_years * 366:
        logging.warning(f"COT start > {max_years}y ago. Truncating to last {max_years}y.")
        start_dt = end_dt - timedelta(days=max_years * 366)

    # batch per year
    batches = []
    ptr = start_dt
    while ptr < end_dt:
        b_end = min(ptr + timedelta(days=365), end_dt)
        batches.append((ptr.strftime("%Y-%m-%d"), b_end.strftime("%Y-%m-%d")))
        ptr = b_end + _DAY

    dfs: List[pd.DataFrame] = []

    def _fetch_batch(s, e):
        where = [f"report_date_as_yyyy_mm_dd >= '{s}'", f"report_date_as_yyyy_mm_dd <= '{e}'"]
        if cot_name:
            where.append(f"market_and_contract_description = '{cot_name}'")
        where_clause = " AND ".join(where)
        try:
            results = _socrata_get_with_retry(client, "6dca-aqww", where_clause, limit=50_000, retries=4, backoff_base=1.5)
            return pd.DataFrame.from_records(results)
        except Exception as exc:
            logging.error(f"COT fetch error {s}→{e}: {exc}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=min(len(batches), 6)) as executor:
        futures = {executor.submit(_fetch_batch, s, e): (s, e) for s, e in batches}
        for fut in as_completed(futures):
            s, e = futures[fut]
            try:
                df_batch = fut.result()
                if df_batch.empty:
                    logging.warning(f"No COT data {s} → {e}")
                else:
                    dfs.append(df_batch)
            except Exception as exc:
                logging.error(f"Batch {s}→{e} raised: {exc}")

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    if "report_date_as_yyyy_mm_dd" not in df_all.columns:
        logging.warning("COT result missing expected date column; returning empty DataFrame.")
        return pd.DataFrame()

    df_all["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df_all["report_date_as_yyyy_mm_dd"])
    num_cols = [c for c in df_all.columns if c != "report_date_as_yyyy_mm_dd"]

    for c in num_cols:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    agg = (
        df_all.groupby("report_date_as_yyyy_mm_dd")[num_cols]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"report_date_as_yyyy_mm_dd": "report_date"})
        .sort_values("report_date")
    )
    return agg