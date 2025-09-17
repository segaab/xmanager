# supabase_logger.py
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import datetime

# Load .env if present (local dev)
load_dotenv()

# Fetch credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found. Please set SUPABASE_URL and SUPABASE_KEY in .env or environment variables.")

class SupabaseLogger:
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def log_metrics(self, model_name: str, epoch: int, loss: float, accuracy: float,
                    expected_return: float = None, realized_return: float = None,
                    hit_rate: float = None, sharpe: float = None):
        """
        Logs model training metrics to the 'training_logs' table in Supabase.

        :param model_name: Name/identifier of the model
        :param epoch: Epoch number
        :param loss: Training loss
        :param accuracy: Training accuracy
        :param expected_return: Expected return (optional)
        :param realized_return: Realized return (optional)
        :param hit_rate: Hit rate (optional)
        :param sharpe: Sharpe ratio (optional)
        """
        data = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model_name": model_name,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "expected_return": expected_return,
            "realized_return": realized_return,
            "hit_rate": hit_rate,
            "sharpe": sharpe
        }
        try:
            response = self.client.table("training_logs").insert(data).execute()
            if response.get("status_code") not in (200, 201):
                print(f"Warning: Failed to insert log to Supabase: {response}")
        except Exception as e:
            print(f"Error logging to Supabase: {e}")