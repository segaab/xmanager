# fetch_data.py
import os
import time
import logging
import pandas as pd
from sodapy import Socrata
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_socrata_client(timeout: int = 60, max_retries: int = 5, backoff_factor: float = 1.5) -> Socrata:
    """
    Initialize Socrata client using environment variables.
    Retries connection in case of failures.
    """
    domain = "publicreporting.cftc.gov"
    app_token = os.getenv("WSCaavlIcDgtLVZbJA1FKkq40")
    username = os.getenv("SEGAB120_EMAIL")
    password = os.getenv("SEGAB120_PASSWORD")
    
    client = None
    for attempt in range(1, max_retries + 1):
        try:
            client = Socrata(domain, app_token, username=username, password=password, timeout=timeout)
            logger.info("Socrata client initialized successfully.")
            return client
        except Exception as e:
            sleep_time = backoff_factor ** attempt
            logger.warning("Socrata client init failed (attempt %d/%d): %s. Retrying in %.1f sec...", attempt, max_retries, e, sleep_time)
            time.sleep(sleep_time)
    raise ConnectionError("Failed to initialize Socrata client after multiple retries.")


def fetch_cot(client: Socrata, dataset_id: str = "6dca-aqww", report_date: Optional[str] = None, max_rows: int = 10000) -> pd.DataFrame:
    """
    Fetch COT data from Socrata API.
    If report_date is None, fetch most recent data.
    """
    where_clause = f"report_date_as_yyyy_mm_dd = '{report_date}'" if report_date else None
    retries = 5
    for attempt in range(1, retries + 1):
        try:
            results = client.get(dataset_id, where=where_clause, limit=max_rows)
            df = pd.DataFrame.from_records(results)
            if not df.empty:
                df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
            return df
        except Exception as e:
            sleep_time = 2 ** attempt
            logger.warning("fetch_cot attempt %d failed: %s. Retrying in %d sec...", attempt, e, sleep_time)
            time.sleep(sleep_time)
    logger.error("fetch_cot failed after %d retries.", retries)
    return pd.DataFrame()


def fetch_price(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV price data for a given symbol from yfinance (or any preferred source)
    """
    import yfinance as yf
    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        return df
    except Exception as e:
        logger.error("fetch_price failed for %s: %s", symbol, e)
        return pd.DataFrame()