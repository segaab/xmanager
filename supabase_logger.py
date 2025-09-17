# supabase_logger.py
import os
from supabase import create_client, Client
from typing import List, Dict, Optional

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
        # Table names
        self.runs_table = "entry_runs"
        self.trades_table = "entry_trades"

    def log_run(self, metrics: Dict, metadata: Dict, trades: Optional[List[Dict]] = None) -> str:
        """
        Log a single pipeline run to Supabase.
        :param metrics: Dict of numeric metrics (accuracy, PnL, win_rate, etc.)
        :param metadata: Dict of run parameters, thresholds, asset info, etc.
        :param trades: Optional list of per-trade dictionaries
        :return: run_id used
        """
        run_id = metadata.get("run_id")
        if not run_id:
            raise ValueError("metadata must include a 'run_id' field")

        # Prepare run record
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

        # Insert into Supabase
        response = self.client.table(self.runs_table).insert(run_record).execute()
        if response.get('error'):
            raise RuntimeError(f"Failed to insert run: {response['error']}")

        # Log trades if provided
        if trades:
            for t in trades:
                t['run_id'] = run_id
            trade_response = self.client.table(self.trades_table).insert(trades).execute()
            if trade_response.get('error'):
                raise RuntimeError(f"Failed to insert trades: {trade_response['error']}")

        return run_id

    def fetch_runs(self, symbol: Optional[str] = None, limit: int = 50):
        """
        Fetch last N runs for an optional symbol filter
        """
        query = self.client.table(self.runs_table)
        if symbol:
            query = query.eq("symbol", symbol)
        query = query.order("start_date", desc=True).limit(limit)
        response = query.execute()
        if response.get('error'):
            raise RuntimeError(f"Failed to fetch runs: {response['error']}")
        return response.get('data', [])

    def fetch_trades(self, run_id: str):
        """
        Fetch trades associated with a specific run_id
        """
        response = self.client.table(self.trades_table).select("*").eq("run_id", run_id).execute()
        if response.get('error'):
            raise RuntimeError(f"Failed to fetch trades: {response['error']}")
        return response.get('data', [])