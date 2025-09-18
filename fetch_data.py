# fetch_data.py
import os
import pandas as pd
from yahooquery import Ticker
from sodapy import Socrata
from datetime import datetime, timedelta

def fetch_price(symbol: str, start: str, end: str, interval: str = "1m"):
    """
    Fetch OHLCV bars via yahooquery.
    start/end format: 'YYYY-MM-DD'
    interval examples: '1m','5m','1h','1d'
    Returns DataFrame indexed by datetime in UTC with columns: open, high, low, close, volume
    """
    tk = Ticker(symbol, asynchronous=True)
    try:
        hist = tk.history(start=start, end=end, interval=interval)
    except Exception as e:
        raise RuntimeError(f"yahooquery fetch error: {e}")

    if isinstance(hist, dict):
        hist = hist.get(symbol, pd.DataFrame())

    if hist.empty:
        return pd.DataFrame()

    df = hist.reset_index().rename(columns={'date':'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    df = df[['open','high','low','close','volume']].copy()
    return df

def init_socrata_client():
    """
    Initialize Socrata client using environment variables:
    SOCRATA_APP_TOKEN, SOCRATA_USERNAME, SOCRATA_PASSWORD
    """
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    client = Socrata("publicreporting.cftc.gov", app_token)
    return client

def fetch_cot(client=None, start=None, end=None, cot_name=None):
    """
    Fetch COT dataset '6dca-aqww' (publicreporting.cftc.gov)
    start/end: 'YYYY-MM-DD' inclusive strings. If None fetch recent ~5 years.
    cot_name: optional, filter by contract_name in the dataset
    Returns DataFrame with report_date (as datetime) and commercial_net/noncom_net etc.
    """
    if client is None:
        client = init_socrata_client()

    where_clauses = []
    if start:
        where_clauses.append(f"report_date_as_yyyy_mm_dd >= '{start}'")
    if end:
        where_clauses.append(f"report_date_as_yyyy_mm_dd <= '{end}'")
    if cot_name:
        where_clauses.append(f"market_and_contract_description = '{cot_name}'")

    where = " AND ".join(where_clauses) if where_clauses else ""

    results = client.get("6dca-aqww", where=where, limit=50000)
    df = pd.DataFrame.from_records(results)
    if df.empty:
        return df

    # canonicalize types
    df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
    numeric_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            numeric_cols.append(col)
        except Exception:
            pass

    # Aggregate by report_date (sum numeric fields)
    agg = df.groupby('report_date_as_yyyy_mm_dd')[numeric_cols].sum().reset_index()
    agg = agg.sort_values('report_date_as_yyyy_mm_dd').rename(columns={'report_date_as_yyyy_mm_dd':'report_date'})
    return agg