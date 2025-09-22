# utils.py
import numpy as np
import pandas as pd

def purged_train_test_split(events_df: pd.DataFrame, purge_radius: int = 10):
    """
    Performs a naive purged train-test split.
    The first 80% of events are assigned to training, the remaining 20% to testing.
    The purge_radius parameter is included for API compatibility with more sophisticated purging schemes.
    """
    n = len(events_df)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    
    split_idx = int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    
    train_mask[:split_idx] = True
    test_mask[split_idx:] = True
    
    return train_mask, test_mask