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

    # Handle dict return (multi-symbol)
    if isinstance(hist, dict):
        hist = hist.get(symbol, pd.DataFrame())

    if hist.empty:
        print(f"Warning: No data returned for {symbol} from {start} to {end} with interval {interval}.")
        return pd.DataFrame()

    # Handle MultiIndex (symbol/date) if present
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index(level=0, drop=True).reset_index()

    # Normalize column names
    df = hist.reset_index().rename(columns={'date':'datetime'})
    if 'datetime' not in df.columns and 'level_0' in df.columns:
        df.rename(columns={'level_0':'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    # Keep OHLCV if exists, otherwise try adjclose as close
    for col in ['open','high','low','close','volume']:
        if col not in df.columns:
            if col == 'close' and 'adjclose' in df.columns:
                df['close'] = df['adjclose']
            else:
                df[col] = 0.0

    df = df[['open','high','low','close','volume']].copy()
    return df

def init_socrata_client():
    """
    Initialize Socrata client using environment variable:
    SOCRATA_APP_TOKEN
    Username/password removed for anonymous access.
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
        print(f"Warning: No COT data returned for {cot_name} between {start} and {end}")
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