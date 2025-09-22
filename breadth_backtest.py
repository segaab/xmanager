# breadth_backtest.py — Breadth Backtest with Logging
import pandas as pd
import logging
from backtest import simulate_limits

logger = logging.getLogger(__name__)

def run_breadth_backtest(clean: pd.DataFrame, bars: pd.DataFrame, symbol: str = "GC=F"):
    """
    Run backtests across different breadth modes (low, mid, high).
    clean : DataFrame with features + labels + predictions
    bars  : Price dataframe (OHLCV)
    symbol: Asset ticker
    """
    results = {}
    logger.info("Starting breadth backtest…")

    if clean is None or clean.empty:
        logger.error("Clean dataset is empty, cannot run backtest.")
        return results

    # Modes
    modes = {
        "low_breadth": {"sell_th": 5, "buy_th": 5},
        "mid_breadth": {"sell_th": 4, "buy_th": 6},
        "high_breadth": {"sell_th": 3, "buy_th": 7},
    }

    for mode, th in modes.items():
        logger.info("Running %s with thresholds: sell<%s, buy>%s", mode, th["sell_th"], th["buy_th"])
        try:
            df = clean.copy()
            df["pred_label"] = 0
            df.loc[df["signal"] > th["buy_th"], "pred_label"] = 1
            df.loc[df["signal"] < th["sell_th"], "pred_label"] = -1

            overlay = simulate_limits(df, bars, label_col="pred_label", symbol=symbol)
            results[mode] = {
                "trades": overlay.shape[0] if overlay is not None else 0,
                "overlay": overlay,
            }
            logger.info("%s completed: %d trades simulated.", mode, results[mode]["trades"])
        except Exception as e:
            logger.error("%s backtest failed: %s", mode, e, exc_info=True)
            results[mode] = {"error": str(e)}

    logger.info("Breadth backtest finished with modes: %s", list(results.keys()))
    return results

# CLI entry
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    print("Run via app.py or import run_breadth_backtest")