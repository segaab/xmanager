# ───────────────────────────────────────── fetch_data.py ─────────────────────────────────────────
"""
Market-data utilities with threaded, batched fetching to support up to 15 years.

Fixes / features:
• Threaded batching for Yahoo intraday & daily bars (up to 15y)
• Threaded batching for COT fetch (yearly slices)
• Error/warning logs for intraday ranges exceeding Yahoo limits
• Light LRU caching for repeated interactive sessions
• Always returns canonical OHLCV dataframe (UTC, tz-aware, deduped)
• COT aggregation corrected
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
    """
    Fetch price bars with threaded batching. Supports up to 15 years.
    """
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    if start_dt >= end_dt:
        raise ValueError("`start` must be before `end`")
    if (end_dt - start_dt).days > max_years * 366:
        logging.warning(f"Requested range > {max_years} years. Truncating to last {max_years} years.")
        start_dt = end_dt - timedelta(days=max_years*366)

    # --- Determine batch size based on interval
    def _max_span(ivl: str) -> timedelta:
        if ivl == "1m":
            return timedelta(days=30)
        if ivl in {"2m","5m","15m","30m","60m","90m","1h"}:
            return timedelta(days=60)
        return timedelta(days=365)  # daily+: safe 1 year per batch for threading

    span = _max_span(interval)
    if _max_span(interval) < (end_dt - start_dt):
        logging.info(f"{interval} range too large for single fetch. Will batch into {span.days}-day chunks.")

    # --- Create batches
    batches = []
    ptr = start_dt
    while ptr < end_dt:
        batch_end = min(ptr + span, end_dt)
        batches.append((ptr.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")))
        ptr = batch_end + _DAY  # avoid overlap

    # --- Threaded fetch
    dfs: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=min(len(batches), 6)) as executor:
        futures = {executor.submit(_fetch_chunk, symbol, s, e, interval): (s,e) for s,e in batches}
        for fut in as_completed(futures):
            s,e = futures[fut]
            try:
                df_chunk = fut.result()
                if df_chunk.empty:
                    logging.warning(f"No data for {symbol} {interval} chunk {s} → {e}")
                dfs.append(df_chunk)
            except Exception as exc:
                logging.error(f"Error fetching chunk {s} → {e}: {exc}")

    if not dfs:
        return pd.DataFrame()

    # --- Canonicalize
    df_raw = pd.concat(dfs, ignore_index=True)
    rename = {
        "date":"datetime","open":"open","high":"high","low":"low","close":"close",
        "adjclose":"adjclose","volume":"volume","Open":"open","High":"high","Low":"low",
        "Close":"close","Adj Close":"adjclose","Volume":"volume"
    }
    df_raw.rename(columns=rename, inplace=True)
    if "datetime" not in df_raw.columns:
        raise RuntimeError("No datetime column found after Yahoo fetch.")

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


# ───────────────────────────────── COT helpers ─────────────────────────────
def init_socrata_client():
    return Socrata("publicreporting.cftc.gov", os.getenv("SOCRATA_APP_TOKEN"))


def fetch_cot(
    client: Socrata | None = None,
    start: str | None = None,
    end: str | None = None,
    cot_name: str | None = None,
    max_years: int = 15,
) -> pd.DataFrame:
    """
    Threaded, batched COT fetch for up to 15 years.
    """
    client = client or init_socrata_client()
    start_dt, end_dt = pd.to_datetime(start) if start else None, pd.to_datetime(end) if end else pd.to_datetime("today")
    if start_dt and (pd.Timestamp.now() - start_dt).days > max_years*366:
        logging.warning(f"COT start > {max_years}y ago. Truncating to last {max_years}y.")
        start_dt = pd.Timestamp.now() - timedelta(days=max_years*366)

    # Batch yearly
    batch_start = start_dt or pd.Timestamp.now() - timedelta(days=365*max_years)
    batch_end = batch_start + timedelta(days=365)
    batches = []
    while batch_start < end_dt:
        b_end = min(batch_start + timedelta(days=365), end_dt)
        batches.append((batch_start.strftime("%Y-%m-%d"), b_end.strftime("%Y-%m-%d")))
        batch_start = b_end + _DAY

    dfs: List[pd.DataFrame] = []
    def _fetch_batch(s,e):
        where = [f"report_date_as_yyyy_mm_dd >= '{s}'", f"report_date_as_yyyy_mm_dd <= '{e}'"]
        if cot_name:
            where.append(f"market_and_contract_description = '{cot_name}'")
        try:
            res = client.get("6dca-aqww", where=" AND ".join(where), limit=50_000)
            return pd.DataFrame.from_records(res)
        except Exception as exc:
            logging.error(f"COT fetch error {s}→{e}: {exc}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=min(len(batches),6)) as executor:
        futures = {executor.submit(_fetch_batch, s,e):(s,e) for s,e in batches}
        for fut in as_completed(futures):
            df_batch = fut.result()
            if df_batch.empty:
                s,e = futures[fut]
                logging.warning(f"No COT data {s} → {e}")
            else:
                dfs.append(df_batch)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df_all["report_date_as_yyyy_mm_dd"])
    num_cols = [c for c in df_all.columns if c != "report_date_as_yyyy_mm_dd"]
    for c in num_cols:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    agg = (
        df_all.groupby("report_date_as_yyyy_mm_dd")[num_cols]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"report_date_as_yyyy_mm_dd":"report_date"})
        .sort_values("report_date")
    )
    return agg