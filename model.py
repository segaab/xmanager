# model.py
"""
Robust XGBoost training & prediction utilities for the confirm-stage model.

Features / improvements:
- Enforces binary labels (maps {-1, 1} → {0,1} and drops NaNs)
- Uses XGBClassifier with a safe base_score (0.5) and early stopping
- Returns a trained model object plus feature list and useful metrics
- Flexible saving options (torch .pt, joblib, or xgboost native save)
- predict_confirm_prob supports both XGBClassifier and raw Booster objects
"""

from __future__ import annotations
import time
import typing as t
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
)
import joblib

try:
    import torch
except Exception:  # torch is optional for saving .pt
    torch = None


def _clean_labels(series: pd.Series) -> pd.Series:
    """Ensure labels are 0 or 1. Map -1/1 -> 0/1, bools -> ints, drop NaNs."""
    s = series.dropna().copy()
    # If labels are strings like "TP"/"SL" user must convert before calling.
    # Handle common numeric encodings:
    if s.dtype.kind in {"i", "u", "f", "b"}:
        # map -1 -> 0, 1 -> 1, 0 -> 0
        s = s.astype(float)
        # If labels are -1/1, map to 0/1
        if set(np.unique(s)).issubset({-1.0, 0.0, 1.0}):
            s = (s > 0).astype(int)
        else:
            # if contains other numeric values, try to threshold at 0
            s = (s > 0).astype(int)
    else:
        # Try to interpret common string labels
        s = s.astype(str).str.strip().str.lower()
        s = s.replace({"tp": 1, "sl": 0, "win": 1, "loss": 0, "true": 1, "false": 0})
        s = pd.to_numeric(s, errors="coerce").dropna().astype(int)
        s = (s > 0).astype(int)
    return s


def train_xgb_confirm(
    candidates: pd.DataFrame,
    feature_cols: t.Sequence[str],
    label_col: str = "label",
    save_path: str | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[xgb.XGBClassifier, list, dict]:
    """
    Train an XGBoost confirm model (binary classifier).

    Returns:
        model: trained XGBClassifier instance
        feature_list: list of feature names used
        metrics: dict with train/val sizes, evaluation metrics, training time, best iteration
    """
    # Input checks
    if len(feature_cols) == 0:
        raise ValueError("feature_cols must contain at least one feature name.")

    if label_col not in candidates.columns:
        raise ValueError(f"Label column '{label_col}' not found in candidates DataFrame.")

    # Prepare data and clean labels
    labels = _clean_labels(candidates[label_col])
    if labels.empty:
        raise ValueError("No valid labels after cleaning (expected labels convertible to 0/1).")

    # Align features with cleaned labels (drop rows without cleaned label)
    df = candidates.loc[labels.index]
    X = df[list(feature_cols)].copy()
    y = labels.loc[X.index].astype(int)

    # Drop rows with NA in features
    mask = ~X.isna().any(axis=1)
    if mask.sum() < len(X):
        if verbose:
            print(f"Dropping {len(X) - mask.sum()} rows with NaNs in features.")
    X = X.loc[mask]
    y = y.loc[mask]

    if X.shape[0] < 10:
        raise ValueError(f"Too few samples for training after cleaning: {X.shape[0]}")

    # Ensure at least two classes present
    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        raise ValueError(f"Only one class present in labels after cleaning: {unique_classes}. Need both classes.")

    # Train / validation split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Build XGBClassifier with safe base_score and sensible defaults
    model = xgb.XGBClassifier(
        n_estimators=int(num_boost_round),
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        base_score=0.5,  # must be in (0,1)
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=int(random_state),
        n_jobs=-1,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds and int(early_stopping_rounds) > 0 else None,
        verbose=verbose,
    )
    train_time = time.time() - t0

    # Predictions on validation
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

    metrics = {}
    try:
        metrics["val_roc_auc"] = float(roc_auc_score(y_val, y_val_pred_proba))
    except Exception:
        metrics["val_roc_auc"] = None
    try:
        metrics["val_logloss"] = float(log_loss(y_val, y_val_pred_proba))
    except Exception:
        metrics["val_logloss"] = None
    metrics["val_accuracy"] = float(accuracy_score(y_val, y_val_pred))
    metrics["val_precision"] = float(precision_score(y_val, y_val_pred, zero_division=0))
    metrics["val_recall"] = float(recall_score(y_val, y_val_pred, zero_division=0))
    metrics["train_time_s"] = float(train_time)
    # best_iteration available on underlying booster after early stopping
    try:
        metrics["best_iteration"] = int(model.get_booster().best_iteration)
    except Exception:
        metrics["best_iteration"] = None
    metrics["train_size"] = int(X_train.shape[0])
    metrics["val_size"] = int(X_val.shape[0])
    metrics["num_features"] = int(X_train.shape[1])
    metrics["feature_cols"] = list(feature_cols)

    # Save model if requested
    if save_path:
        save_path = str(save_path)
        if save_path.lower().endswith(".pt") and torch is not None:
            # torch.save the dict (keeps Python object)
            torch.save({"model": model, "features": list(feature_cols)}, save_path)
        elif save_path.lower().endswith((".pkl", ".joblib")):
            joblib.dump({"model": model, "features": list(feature_cols)}, save_path)
        else:
            # fallback to xgboost native save (JSON) for booster
            try:
                model.get_booster().save_model(save_path)
            except Exception:
                # as final fallback save via joblib
                joblib.dump({"model": model, "features": list(feature_cols)}, save_path)

    return model, list(feature_cols), metrics


def predict_confirm_prob(model_obj, candidates: pd.DataFrame, feature_cols: t.Sequence[str]) -> pd.Series:
    """
    Predict confirm probabilities for candidate events.

    Accepts either:
      - XGBClassifier (scikit-learn wrapper) or
      - xgboost.Booster (raw booster)

    Returns: pandas Series of probabilities aligned with candidates.index named 'confirm_prob'.
    """
    if candidates.empty:
        return pd.Series([], index=candidates.index, name="confirm_prob")

    X = candidates[list(feature_cols)].copy()
    if X.isna().any(axis=1).any():
        # drop NA rows (user code should ensure features present)
        X = X.dropna(axis=0, how="any")
        if X.empty:
            return pd.Series([], index=candidates.index, name="confirm_prob")

    # XGBClassifier
    if isinstance(model_obj, xgb.XGBClassifier):
        probs = model_obj.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index, name="confirm_prob")

    # raw Booster
    if isinstance(model_obj, xgb.Booster):
        dmatrix = xgb.DMatrix(X, feature_names=list(feature_cols))
        probs = model_obj.predict(dmatrix)
        return pd.Series(probs, index=X.index, name="confirm_prob")

    # If saved dict (e.g., torch/joblib saved), try to extract
    if isinstance(model_obj, dict):
        inner = model_obj.get("model") or model_obj.get("booster")
        if inner is not None:
            return predict_confirm_prob(inner, candidates, feature_cols)

    raise TypeError("Unsupported model object type for prediction. Provide XGBClassifier, xgboost.Booster, or dict.")