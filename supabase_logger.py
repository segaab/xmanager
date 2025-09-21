# supabase_logger.py
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseLogger:
    def __init__(self):
        """
        Initialize Supabase client using environment variables:
        SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
        """
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("Supabase URL or service key not found in environment variables.")
        self.client: Client = create_client(url, key)
        self.runs_table = "entry_runs"
        self.trades_table = "entry_trades"

    def log_run(self, metrics: Dict, metadata: Dict, trades: Optional[List[Dict]] = None) -> str:
        """
        Insert a run record into the runs table, optionally log trades linked to the run_id.
        """
        run_id = metadata.get("run_id")
        if not run_id:
            raise ValueError("metadata must include a 'run_id' field")

        run_record = {
            "run_id": run_id,
            "symbol": metadata.get("symbol"),
            "start_date": metadata.get("start_date"),
            "end_date": metadata.get("end_date"),
            "interval": metadata.get("interval"),
            "feature_cols": metadata.get("feature_cols"),
            "model_file": metadata.get("model_file"),
            "training_params": metadata.get("training_params"),
            "health_thresholds": metadata.get("health_thresholds"),
            "p_fast": metadata.get("p_fast"),
            "p_slow": metadata.get("p_slow"),
            "p_deep": metadata.get("p_deep"),
            "metrics": metrics,
        }

        resp = self.client.table(self.runs_table).insert(run_record).execute()
        if getattr(resp, "error", None):
            logger.error("Failed to insert run: %s", resp.error)
            raise RuntimeError(f"Failed to insert run: {resp.error}")
        logger.info("Inserted run %s into %s", run_id, self.runs_table)

        if trades:
            sanitized = []
            for t in trades:
                t_copy = {}
                for k, v in t.items():
                    if isinstance(v, (pd.Timestamp, datetime)):
                        t_copy[k] = v.isoformat()
                    else:
                        t_copy[k] = v
                t_copy["run_id"] = run_id
                sanitized.append(t_copy)

            tr_resp = self.client.table(self.trades_table).insert(sanitized).execute()
            if getattr(tr_resp, "error", None):
                logger.error("Failed to insert trades: %s", tr_resp.error)
                raise RuntimeError(f"Failed to insert trades: {tr_resp.error}")
            logger.info("Inserted %d trades for run %s", len(sanitized), run_id)

        return run_id

    def fetch_runs(self, symbol: Optional[str] = None, limit: int = 50):
        """
        Fetch recent run records, optionally filtered by symbol.
        """
        query = self.client.table(self.runs_table)
        if symbol:
            query = query.eq("symbol", symbol)
        resp = query.order("start_date", desc=True).limit(limit).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch runs: {resp.error}")
        return getattr(resp, "data", [])

    def fetch_trades(self, run_id: str):
        """
        Fetch all trades linked to a specific run_id.
        """
        resp = self.client.table(self.trades_table).select("*").eq("run_id", run_id).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch trades: {resp.error}")
        return getattr(resp, "data", [])