# fetch_data.py – updated fetch_price with threading & batching
from concurrent.futures import ThreadPoolExecutor, as_completed
from yahooquery import Ticker
import pandas as pd
import numpy as np
import logging
import time
from datetime import timedelta, datetime

_DAY = timedelta(days=1)

def _fetch_price_chunk(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
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

def fetch_price(symbol: str, start: str, end: str, interval: str = "1d", max_years: int = 15, max_workers: int = 6) -> pd.DataFrame:
    """
    Fetch price data in threaded, batched chunks.
    """
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    if start_dt >= end_dt:
        raise ValueError("`start` must be before `end`")
    if (end_dt - start_dt).days > max_years * 366:
        logging.warning(f"Requested range > {max_years} years. Truncating to last {max_years} years.")
        start_dt = end_dt - timedelta(days=max_years*366)

    # Determine chunk size
    if interval == "1m":
        chunk_days = 30
    elif interval in {"2m","5m","15m","30m","60m","90m","1h"}:
        chunk_days = 60
    else:
        chunk_days = 365  # daily+

    # Create batch intervals
    batches = []
    ptr = start_dt
    while ptr < end_dt:
        b_end = min(ptr + timedelta(days=chunk_days), end_dt)
        batches.append((ptr.strftime("%Y-%m-%d"), b_end.strftime("%Y-%m-%d")))
        ptr = b_end + _DAY

    # Threaded fetch
    dfs = []
    with ThreadPoolExecutor(max_workers=min(len(batches), max_workers)) as executor:
        futures = {executor.submit(_fetch_price_chunk, symbol, s, e, interval): (s, e) for s, e in batches}
        for fut in as_completed(futures):
            s, e = futures[fut]
            try:
                df_chunk = fut.result()
                if df_chunk.empty:
                    logging.warning(f"No data fetched for {symbol} {s} → {e}")
                else:
                    dfs.append(df_chunk)
            except Exception as exc:
                logging.error(f"Chunk {s} → {e} failed: {exc}")
            time.sleep(0.15)  # slight delay to avoid Yahoo throttling

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # Standardize columns
    rename = {
        "date":"datetime","open":"open","high":"high","low":"low","close":"close",
        "adjclose":"adjclose","volume":"volume","Open":"open","High":"high","Low":"low",
        "Close":"close","Adj Close":"adjclose","Volume":"volume"
    }
    df_all.rename(columns=rename, inplace=True)
    if "datetime" not in df_all.columns:
        raise RuntimeError("Could not locate a datetime column in yahooquery output.")
    if "close" not in df_all.columns and "adjclose" in df_all.columns:
        df_all["close"] = df_all["adjclose"]

    keep = ["open","high","low","close","volume"]
    for col in keep:
        if col not in df_all.columns:
            df_all[col] = np.nan

    out = (
        df_all.set_index(pd.to_datetime(df_all["datetime"], utc=True))
        .sort_index()
        .loc[:, keep]
        .dropna(how="all")
    )
    out = out[~out.index.duplicated(keep="first")]
    return out