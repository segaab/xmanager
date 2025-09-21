# features.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute rolling relative volume (RVol) for the 'volume' column.
    Returns pd.Series aligned with df.index.
    """
    if "volume" not in df.columns:
        raise KeyError("DataFrame must contain a 'volume' column to compute RVol")

    rvol = df["volume"].rolling(window=window, min_periods=1).mean() / df["volume"].rolling(window=window * 2, min_periods=1).mean()
    rvol = rvol.fillna(0.0)
    return rvol


def compute_health_gauge(df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """
    Compute a simple HealthGauge score using available micro-features.
    weights: dict mapping feature name -> weight. Defaults applied if None.
    """
    default_weights = {
        "rvol": 0.4,
        "uptick_ratio": 0.2,
        "buy_vol_ratio": 0.2,
        "micro_range": 0.2
    }
    if weights is None:
        weights = default_weights
    else:
        # ensure all required features present
        for k, v in default_weights.items():
            if k not in weights:
                weights[k] = v

    # ensure features exist in df
    for feat in weights.keys():
        if feat not in df.columns:
            logger.warning("Feature '%s' missing in DataFrame, filling with 0.0", feat)
            df[feat] = 0.0

    score = sum(df[feat].fillna(0.0) * w for feat, w in weights.items())
    return pd.Series(score, index=df.index, name="health_gauge")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure RVol and HealthGauge exist, no duplicate index.
    """
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df["rvol"] = compute_rvol(df)
    df["health_gauge"] = compute_health_gauge(df)
    return df