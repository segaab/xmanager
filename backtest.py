# backtest.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def backtest_candidates(candidates: pd.DataFrame, initial_capital: float = 10000.0):
    """
    Backtest trade candidates and compute stats.
    Adds open_time / close_time and ensures candidate_time is robust.
    """
    if candidates is None or candidates.empty:
        logger.warning("No candidates provided for backtest.")
        return pd.DataFrame()

    df = candidates.copy()

    # Ensure datetime index for candidate_time
    if "candidate_time" not in df.columns:
        df["candidate_time"] = df.index
    df["candidate_time"] = pd.to_datetime(df["candidate_time"], utc=True)

    # Ensure end_time exists and is timezone-aware
    if "end_time" not in df.columns:
        df["end_time"] = df["candidate_time"] + pd.to_timedelta(df.get("duration", 0), unit="m")
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    # Open/close times for overlay plots
    df["open_time"] = df["candidate_time"]
    df["close_time"] = df["end_time"]

    # Compute returns and equity
    df["pnl"] = df["realized_return"] * initial_capital
    df["equity"] = initial_capital + df["pnl"].cumsum()

    # Trade statistics
    stats = {
        "total_trades": len(df),
        "win_trades": int((df["realized_return"] > 0).sum()),
        "loss_trades": int((df["realized_return"] <= 0).sum()),
        "win_rate": float((df["realized_return"] > 0).mean()),
        "avg_return": float(df["realized_return"].mean()),
        "max_drawdown": float((df["equity"].cummax() - df["equity"]).max()),
        "final_equity": float(df["equity"].iloc[-1]),
    }
    logger.info("Backtest stats: %s", stats)

    return df, stats