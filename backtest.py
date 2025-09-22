# backtest.py — Core Backtesting Logic with Logging
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def simulate_limits(
    df: pd.DataFrame,
    bars: pd.DataFrame,
    label_col: str = "pred_label",
    symbol: str = "GC=F",
    rr: float = 2.0,
    sl: float = 0.01,
    tp: float = 0.02,
    max_holding: int = 20
) -> pd.DataFrame:
    """
    Simulate trades given entry signals and OHLC bars.

    Parameters
    ----------
    df : DataFrame
        Candidate trades with entry times and signals
    bars : DataFrame
        OHLC price data
    label_col : str
        Column name in df indicating long/short signals
    symbol : str
        Asset ticker
    rr : float
        Risk/reward ratio
    sl : float
        Stop-loss percentage
    tp : float
        Take-profit percentage
    max_holding : int
        Maximum bars to hold a position

    Returns
    -------
    overlay : DataFrame
        Simulated trade results with pnl, entry/exit info
    """
    logger.info("Starting simulate_limits for %s…", symbol)

    if df is None or df.empty:
        logger.warning("Input df is empty — returning empty results.")
        return pd.DataFrame()

    if bars is None or bars.empty:
        logger.error("Bars data is missing — cannot run backtest.")
        return pd.DataFrame()

    trades = []
    for idx, row in df.iterrows():
        if row.get(label_col, 0) == 0:
            continue

        entry_time = row.get("time") or row.name
        if entry_time not in bars.index:
            logger.debug("Entry time %s not found in bars index — skipping.", entry_time)
            continue

        entry_price = bars.loc[entry_time, "close"]
        direction = int(row[label_col])

        # Set stop-loss and take-profit levels
        sl_price = entry_price * (1 - sl) if direction > 0 else entry_price * (1 + sl)
        tp_price = entry_price * (1 + tp) if direction > 0 else entry_price * (1 - tp)

        exit_time, exit_price, pnl = None, None, None
        holding_bars = bars.loc[entry_time:].head(max_holding)

        for t, b in holding_bars.iterrows():
            if direction > 0:
                if b["low"] <= sl_price:  # stop hit
                    exit_time, exit_price, pnl = t, sl_price, -sl
                    break
                if b["high"] >= tp_price:  # tp hit
                    exit_time, exit_price, pnl = t, tp_price, tp
                    break
            else:
                if b["high"] >= sl_price:  # stop hit
                    exit_time, exit_price, pnl = t, sl_price, -sl
                    break
                if b["low"] <= tp_price:  # tp hit
                    exit_time, exit_price, pnl = t, tp_price, tp
                    break

        if exit_time is None:
            exit_time = holding_bars.index[-1]
            exit_price = holding_bars.iloc[-1]["close"]
            pnl = (exit_price - entry_price) / entry_price * direction

        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": direction,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "pnl": pnl
        })

    overlay = pd.DataFrame(trades)

    if overlay.empty:
        logger.warning("No trades were generated for %s.", symbol)
    else:
        logger.info("simulate_limits finished: %d trades generated, avg pnl=%.4f",
                    len(overlay), overlay["pnl"].mean())

    return overlay


# CLI entry
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    print("Run via app.py or import simulate_limits")