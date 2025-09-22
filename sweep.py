# sweep.py — Sweep Mode with Logging
import pandas as pd
import logging
from backtest import simulate_limits

logger = logging.getLogger(__name__)

def run_sweep(clean: pd.DataFrame, bars: pd.DataFrame, symbol: str = "GC=F"):
    """
    Run sweeps across multiple parameter configurations.
    clean : DataFrame with features + labels + predictions
    bars  : Price dataframe (OHLCV)
    """
    results = {}
    logger.info("Starting sweep mode backtest…")

    if clean is None or clean.empty:
        logger.error("Clean dataset is empty, cannot run sweep.")
        return results

    # Sweep parameters
    sell_thresholds = [3, 4, 5]
    buy_thresholds = [5, 6, 7]

    for s in sell_thresholds:
        for b in buy_thresholds:
            key = f"sell{s}_buy{b}"
            logger.info("Sweep iteration %s: sell<%d, buy>%d", key, s, b)
            try:
                df = clean.copy()
                df["pred_label"] = 0
                df.loc[df["signal"] > b, "pred_label"] = 1
                df.loc[df["signal"] < s, "pred_label"] = -1

                overlay = simulate_limits(df, bars, label_col="pred_label", symbol=symbol)
                results[key] = {
                    "trades": overlay.shape[0] if overlay is not None else 0,
                    "overlay": overlay,
                }
                logger.info("Sweep %s completed: %d trades simulated.", key, results[key]["trades"])
            except Exception as e:
                logger.error("Sweep %s failed: %s", key, e, exc_info=True)
                results[key] = {"error": str(e)}

    logger.info("Sweep backtest finished with %d iterations.", len(results))
    return results

# CLI entry
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    print("Run via app.py or import run_sweep")