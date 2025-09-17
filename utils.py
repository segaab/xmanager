# utils.py
import numpy as np
import pandas as pd

def purged_train_test_split(events_df, purge_radius=10):
    """
    Very small helper: naive purged split by index. For demo only.
    events_df indexed by datetime; returns boolean mask arrays.
    """
    n = len(events_df)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    split = int(n*0.8)
    train_mask[:split] = True
    test_mask[split:] = True
    return train_mask, test_mask