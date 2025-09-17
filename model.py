# model.py
"""
XGBoost confirm model utilities.

- train_xgb_confirm: trains an XGBoost booster (default 500 rounds, with early stopping).
- predict_confirm_prob: robust predictor that accepts a trained xgboost.Booster
  or the saved joblib dict ({'model':booster,'features':[...]}) produced by train_xgb_confirm.
"""

from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import xgboost as xgb


def train_xgb_confirm(
    df_features: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "label",
    save_path: Optional[str] = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[xgb.Booster, List[str], dict]:
    """
    Train an XGBoost binary classifier for confirm probability.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame containing features and label_col.
    feature_cols : list[str], optional
        List of column names to use as features. If None, infer automatically.
    label_col : str
        Column name for label (binary 0/1).
    save_path : str, optional
        If provided, saves {'model':booster,'features':feature_cols} to this path via joblib.
    num_boost_round : int
        Number of boosting rounds (default 500).
    early_stopping_rounds : int
        Early stopping on validation set.
    test_size : float
        Fraction of data held out as test (simple split; replace with purged CV for production).
    random_state : int
    verbose : bool
        If True prints eval messages.

    Returns
    -------
    model : xgb.Booster
    feature_cols : list[str]
    metrics : dict (auc, precision, recall, f1)
    """
    if feature_cols is None:
        feature_cols = [c for c in df_features.columns if c not in (label_col, "t_idx", "ret")]

    # Prepare X/y
    df = df_features.copy()
    X = df[feature_cols].fillna(0).values
    y = df[label_col].astype(int).values

    # Simple time-ordered train/test split (no shuffle). Replace with purged CV for real experiments.
    split_idx = int((1 - test_size) * len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state,
        "verbosity": 0,
    }

    evals = [(dtest, "validation")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
    )

    # Evaluate on test set
    preds_proba = booster.predict(dtest)
    try:
        auc = float(roc_auc_score(y_test, preds_proba))
    except Exception:
        auc = float("nan")
    preds_bin = (preds_proba > 0.6).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds_bin, average="binary", zero_division=0)
    metrics = {"auc": auc, "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    if save_path:
        joblib.dump({"model": booster, "features": feature_cols}, save_path)

    return booster, feature_cols, metrics


def predict_confirm_prob(
    model: Union[xgb.Booster, dict],
    features_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Predict confirm probabilities.

    model : xgb.Booster or dict
        If dict, expects {'model':booster,'features':[...]} as saved by train_xgb_confirm.
    features_df : pd.DataFrame
        DataFrame containing feature columns.
    feature_cols : list[str], optional
        If model is a Booster, provide feature_cols that match the training order.

    Returns
    -------
    probs : np.ndarray (shape (n_samples,))
    """
    if isinstance(model, dict):
        booster = model.get("model")
        cols = model.get("features")
    else:
        booster = model
        cols = feature_cols

    if booster is None or cols is None:
        raise ValueError("Model or feature columns not provided / invalid model dict.")

    # Ensure features_df contains required columns
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = features_df[cols].fillna(0).values
    dmat = xgb.DMatrix(X)
    probs = booster.predict(dmat)
    return probs