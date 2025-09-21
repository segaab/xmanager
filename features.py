# features.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Compute relative volume (RVOL) as rolling ratio of current volume to average volume over lookback.
    Returns a pd.Series aligned with df.index.
    """
    if "volume" not in df.columns:
        logger.warning("compute_rvol: 'volume' column missing, returning zeros")
        return pd.Series(0.0, index=df.index)

    rolling_avg = df["volume"].rolling(window=lookback, min_periods=1).mean()
    rvol = df["volume"] / rolling_avg.replace(0, np.nan)
    rvol = rvol.fillna(1.0)  # default to 1.0 if NaN
    return rvol


def calculate_health_gauge(df: pd.DataFrame, rvol_col: str = "rvol", threshold: float = 1.5) -> pd.Series:
    """
    Calculate HealthGauge score for candidate filtering.
    - Example: score = 1 if RVOL above threshold, else 0
    Returns pd.Series aligned with df.index.
    """
    if rvol_col not in df.columns:
        logger.warning("calculate_health_gauge: '%s' column missing, returning zeros", rvol_col)
        return pd.Series(0, index=df.index)

    score = (df[rvol_col] >= threshold).astype(int)
    return score


def ensure_no_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure index is unique and sorted.
    """
    if df.index.duplicated().any():
        logger.warning("Duplicate indices detected; keeping first occurrence")
        df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df