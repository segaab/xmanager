# summary.py — Aggregation of Backtest Results with Logging
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize trades into performance statistics.

    Parameters
    ----------
    trades : DataFrame
        Must contain columns: ["symbol","entry_time","exit_time","pnl"]

    Returns
    -------
    summary : DataFrame
        Aggregate performance metrics
    """
    if trades is None or trades.empty:
        logger.warning("summarize_trades called with empty trades DataFrame")
        return pd.DataFrame()

    logger.info("Summarizing %d trades…", len(trades))

    summary = pd.DataFrame([{
        "total_trades": len(trades),
        "win_rate": (trades["pnl"] > 0).mean(),
        "avg_pnl": trades["pnl"].mean(),
        "median_pnl": trades["pnl"].median(),
        "total_pnl": trades["pnl"].sum(),
        "max_drawdown": trades["pnl"].cumsum().min(),
        "start_time": trades["entry_time"].min(),
        "end_time": trades["exit_time"].max()
    }])

    logger.info(
        "Summary: %d trades | win_rate=%.2f | total_pnl=%.4f",
        summary["total_trades"].iloc[0],
        summary["win_rate"].iloc[0],
        summary["total_pnl"].iloc[0]
    )

    return summary


def combine_summaries(results: dict) -> pd.DataFrame:
    """
    Combine multiple trade summaries into one DataFrame.

    Parameters
    ----------
    results : dict
        {mode: trades_df}

    Returns
    -------
    combined : DataFrame
        Summary by mode
    """
    if not results:
        logger.warning("combine_summaries called with empty results dict")
        return pd.DataFrame()

    logger.info("Combining summaries for %d modes…", len(results))
    combined = []

    for mode, trades_df in results.items():
        if trades_df is None or trades_df.empty:
            logger.debug("Mode %s has no trades", mode)
            continue

        s = summarize_trades(trades_df)
        s["mode"] = mode
        combined.append(s)

    if not combined:
        logger.warning("No valid trades in results dict — combined summary is empty")
        return pd.DataFrame()

    combined_df = pd.concat(combined, ignore_index=True)
    logger.info("combine_summaries finished with %d rows", len(combined_df))

    return combined_df