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
    Returns DataFrame indexed by datetime in UTC with columns: open,high,low,close,volume
    """
    tk = Ticker(symbol, asynchronous=True)
    # yahooquery uses timezone-aware datetimes; ensure inputs
    try:
        hist = tk.history(start=start, end=end, interval=interval)
    except Exception as e:
        raise RuntimeError(f"yahooquery fetch error: {e}")

    if isinstance(hist, dict):
        # sometimes returns dict keyed by symbol
        hist = hist.get(symbol, pd.DataFrame())

    if hist.empty:
        return pd.DataFrame()

    # Normalize
    df = hist.reset_index().rename(columns={'date':'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()
    # keep standard cols
    df = df[['open','high','low','close','volume']].copy()
    return df

def init_socrata_client():
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    username = os.getenv("SOCRATA_USERNAME")
    password = os.getenv("SOCRATA_PASSWORD")
    # If app_token is None, Socrata allows anonymous but rate-limited
    client = Socrata("publicreporting.cftc.gov", app_token,
                     username=username, password=password)
    return client

def fetch_cot(client=None, start=None, end=None):
    """
    Fetch COT dataset '6dca-aqww' (publicreporting.cftc.gov)
    start/end: 'YYYY-MM-DD' inclusive strings. If None fetch recent ~5 years.
    Returns DataFrame with report_date (as datetime) and commercial_net/noncom_net etc.
    """
    if client is None:
        client = init_socrata_client()

    where = ""
    if start and end:
        where = f"report_date_as_yyyy_mm_dd between '{start}' and '{end}'"
    elif start:
        where = f"report_date_as_yyyy_mm_dd >= '{start}'"
    elif end:
        where = f"report_date_as_yyyy_mm_dd <= '{end}'"

    results = client.get("6dca-aqww", where=where, limit=50000)
    df = pd.DataFrame.from_records(results)
    if df.empty:
        return df
    # canonicalize types
    df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
    # the dataset contains many rows for different contracts. For demo we'll aggregate by report_date
    # using net non-commercial position as a proxy: sum across contracts
    numeric_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            numeric_cols.append(col)
        except Exception:
            pass
    agg = df.groupby('report_date_as_yyyy_mm_dd')[numeric_cols].sum().reset_index()
    agg = agg.sort_values('report_date_as_yyyy_mm_dd').rename(columns={'report_date_as_yyyy_mm_dd':'report_date'})
    return agg