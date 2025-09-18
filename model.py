# model.py
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def train_xgb_confirm(
    candidates: pd.DataFrame,
    feature_cols,
    label_col='label',
    save_path=None,
    num_boost_round=500,
    early_stopping_rounds=20,
    test_size=0.2,
    random_state=42,
    verbose=False,
):
    """
    Train an XGBoost model for confirm classification (binary).
    Returns: booster, feature_list, metrics dict
    """

    # --- Clean labels ---
    candidates = candidates.dropna(subset=[label_col])
    candidates = candidates[candidates[label_col].isin([0, 1])]
    if candidates.empty:
        raise ValueError("No valid labels found (must be 0 or 1).")

    X = candidates[feature_cols].values
    y = candidates[label_col].astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",   # ✅ Binary classification
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state,
    }

    evals = [(dtrain, "train"), (dval, "eval")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
    )

    # Collect metrics
    metrics = {
        "best_iteration": booster.best_iteration,
        "best_score": booster.best_score,
        "num_features": len(feature_cols),
        "train_size": len(y_train),
        "val_size": len(y_val),
    }

    # Save model if requested
    if save_path:
        torch.save(
            {"model": booster, "features": feature_cols},
            save_path
        )

    return booster, feature_cols, metrics


def predict_confirm_prob(booster, candidates: pd.DataFrame, feature_cols):
    """
    Predict confirm probabilities for candidate events.
    Returns a pandas Series aligned with candidates index.
    """
    if candidates.empty:
        return pd.Series([], index=candidates.index)

    X = candidates[feature_cols].values
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    preds = booster.predict(dmat)
    return pd.Series(preds, index=candidates.index, name="confirm_prob")