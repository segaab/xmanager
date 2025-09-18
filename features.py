# features.py
import pandas as pd
import numpy as np

def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute Relative Volume (RVol) over rolling window.
    Adds 'rvol' column to df.
    """
    df = df.copy()
    df['volume_avg'] = df['volume'].rolling(window=window, min_periods=1).mean()
    df['rvol'] = df['volume'] / df['volume_avg']
    df['rvol'] = df['rvol'].fillna(1.0)
    return df

def calculate_health_gauge(cot_df: pd.DataFrame, daily_bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HealthGauge combining COT and price/volume proxies.
    Returns dataframe with 'health_gauge' [0,1] column indexed by date.
    """
    df = daily_bars.copy()
    df['health_gauge'] = 0.5  # default neutral

    if not cot_df.empty:
        # normalize commercial net positions vs total
        cot_df = cot_df.sort_values('report_date')
        cot_df['com_net_norm'] = (cot_df.get('commercial_long',0) - cot_df.get('commercial_short',0)) / (cot_df.get('commercial_long',1) + cot_df.get('commercial_short',1))
        cot_df['com_net_norm'] = cot_df['com_net_norm'].clip(-1,1)
        # interpolate onto daily bars
        cot_interp = cot_df.set_index('report_date')['com_net_norm'].reindex(df.index).interpolate(method='time').fillna(0)
        df['health_gauge'] += 0.5 * cot_interp  # weight COT at 50%

    # Clip to [0,1]
    df['health_gauge'] = df['health_gauge'].clip(0,1)
    return df[['health_gauge']]