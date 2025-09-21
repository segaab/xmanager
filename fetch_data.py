# fetch_data.py
import os
import time
import logging
import pandas as pd
from sodapy import Socrata
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Socrata client from environment variables
APP_TOKEN = os.getenv("WSCaavlIcDgtLVZbJA1FKk40")
USERNAME = os.getenv("SEGAB_USER")
PASSWORD = os.getenv("SEGAB_PASS")
BASE_URL = "publicreporting.cftc.gov"
DATASET_ID = "6dca-aqww"

client = Socrata(BASE_URL, APP_TOKEN, username=USERNAME, password=PASSWORD)


def fetch_cot_data(start_date: Optional[str] = None, end_date: Optional[str] = None, retries: int = 5, backoff: float = 2.0) -> pd.DataFrame:
    """
    Fetch COT dataset from Socrata with retry/backoff.
    Parameters:
    - start_date, end_date: strings in 'YYYY-MM-DD' format for filtering report_date_as_yyyy_mm_dd
    - retries: number of retry attempts
    - backoff: initial backoff seconds, doubles each retry
    Returns: pandas DataFrame
    """
    where_clause = ""
    if start_date:
        where_clause += f"report_date_as_yyyy_mm_dd >= '{start_date}'"
    if end_date:
        if where_clause:
            where_clause += f" AND "
        where_clause += f"report_date_as_yyyy_mm_dd <= '{end_date}'"

    attempt = 0
    while attempt < retries:
        try:
            results = client.get(DATASET_ID, where=where_clause) if where_clause else client.get(DATASET_ID)
            df = pd.DataFrame.from_records(results)
            logger.info("Fetched %d records from Socrata", len(df))
            return df
        except Exception as e:
            wait = backoff * (2 ** attempt)
            logger.warning("Fetch attempt %d failed: %s. Retrying in %.1f sec...", attempt + 1, e, wait)
            time.sleep(wait)
            attempt += 1

    raise RuntimeError(f"Failed to fetch data from Socrata after {retries} attempts")