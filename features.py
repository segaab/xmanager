# features.py
import numpy as np
import pandas as pd

def compute_rvol(bars: pd.DataFrame, window: int = 20, vol_col: str = 'volume'):
    """
    rvol = volume / rolling_mean(volume, window)
    bars: DataFrame indexed by datetime with 'volume'
    """
    df = bars.copy()
    df['rvol'] = df[vol_col] / (df[vol_col].rolling(window).mean().replace(0, np.nan))
    df['rvol'] = df['rvol'].fillna(0)
    return df

def calculate_health_gauge(cot_df: pd.DataFrame, daily_bars: pd.DataFrame):
    """
    Simple deterministic HealthGauge combining:
      - change in non-commercial net positions (d_noncom)
      - open interest change (if available) as proxy
      - recent rvol & vol trend
    Returns daily_health_gauge DataFrame (indexed by date)
    """
    # Precondition: cot_df has 'report_date' and numeric aggregated fields.
    # For demo we search common fields; otherwise user may map their own names.
    cot = cot_df.copy().set_index('report_date').sort_index()
    cot = cot.rename_axis('date')

    # forward-fill cot to daily index (align with daily_bars)
    daily = daily_bars.copy()
    daily_index = pd.to_datetime(daily.index.date)
    daily_dates = pd.to_datetime(daily.index.normalize()).unique()
    # Build a daily date index
    df_daily = daily.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(how='all')
    df_daily.index = pd.to_datetime(df_daily.index)
    # Forward-fill COT values
    cot_ff = cot.reindex(df_daily.index, method='ffill')
    # choose a non-commercial column if present
    possible_noncom = [c for c in cot.columns if 'non' in c.lower() and 'commercial' not in c.lower()]
    noncom = cot[possible_noncom[0]] if possible_noncom else cot.iloc[:,0]
    noncom = noncom.reindex(df_daily.index, method='ffill').fillna(0)

    # open interest proxy: use total_open_interest or sum of some OI cols if exist
    possible_oi = [c for c in cot.columns if 'open' in c.lower() and 'interest' in c.lower()]
    if possible_oi:
        oi = cot[possible_oi[0]].reindex(df_daily.index, method='ffill').fillna(0)
    else:
        # fallback: use volume-based proxy from daily bars
        oi = df_daily['volume'].rolling(7).mean().fillna(0)

    # compute signals
    d_noncom = noncom.diff(1).fillna(0)
    d_oi = oi.diff(1).fillna(0)
    # volatility proxy
    df_daily['ret'] = df_daily['close'].pct_change().fillna(0)
    vol_10 = df_daily['ret'].rolling(10).std().fillna(0)
    rvol = df_daily['volume'] / df_daily['volume'].rolling(20).mean().replace(0,np.nan)
    rvol = rvol.fillna(0)

    # zscore each component
    def z(x):
        return (x - x.rolling(60, min_periods=1).mean()) / (x.rolling(60, min_periods=1).std().replace(0, np.nan))

    z_non = z(d_noncom.fillna(0)).fillna(0)
    z_oi = z(d_oi.fillna(0)).fillna(0)
    z_vol = z(vol_10.fillna(0)).fillna(0)
    z_rvol = z(rvol.fillna(0)).fillna(0)

    # Combine with heuristic weights
    gauge_raw = 0.5 * z_non + 0.2 * z_oi - 0.3 * z_vol + 0.4 * z_rvol
    # scale to 0..1 via tanh
    gauge_scaled = np.tanh(gauge_raw / (np.nanstd(gauge_raw) + 1e-8))
    # normalize to [0,1]
    gauge_norm = (gauge_scaled - np.nanmin(gauge_scaled)) / (np.nanmax(gauge_scaled) - np.nanmin(gauge_scaled) + 1e-8)

    df_out = pd.DataFrame({
        'close': df_daily['close'],
        'rvol': rvol,
        'vol_10': vol_10,
        'd_noncom': d_noncom,
        'd_oi': d_oi,
        'health_raw': gauge_raw,
        'health_gauge': gauge_norm
    }, index=df_daily.index)

    return df_out