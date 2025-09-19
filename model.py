# model.py
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BoosterWrapper:
    """
    Thin sklearn-like wrapper around an xgboost.Booster trained with xgb.train.
    Provides predict_proba and predict to behave like sklearn estimators.
    """
    def __init__(self, booster: xgb.Booster, feature_names: List[str]):
        self.booster = booster
        self.feature_names = feature_names
        # best_iteration may or may not exist depending on training call
        self.best_iteration = getattr(booster, "best_iteration", None)

    def _dmatrix(self, X: pd.DataFrame):
        # ensure columns ordering
        if isinstance(X, pd.DataFrame):
            Xp = X[self.feature_names].copy()
        else:
            # assume array-like, convert to DataFrame with feature_names
            Xp = pd.DataFrame(X, columns=self.feature_names)
        return xgb.DMatrix(Xp, feature_names=self.feature_names)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dmat = self._dmatrix(X)
        if self.best_iteration is not None:
            raw = self.booster.predict(dmat, iteration_range=(0, int(self.best_iteration) + 1))
        else:
            raw = self.booster.predict(dmat)
        # raw are probabilities for positive class (binary:logistic)
        raw = np.asarray(raw, dtype=float)
        # return shape (n_samples, 2) like sklearn: [prob_neg, prob_pos]
        probs = np.vstack([1.0 - raw, raw]).T
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def train_xgb_confirm(
    clean: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
    num_boost_round: int = 200,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Train a binary confirm model using xgboost.train (robust to sklearn-fit kwargs differences).
    Returns (model_wrapper, feature_cols, metrics).
    """
    # --------------------- validation & sanitization -------------------------
    missing = [c for c in feature_cols if c not in clean.columns]
    if missing:
        raise KeyError(f"Missing feature columns in training data: {missing}")
    if label_col not in clean.columns:
        raise KeyError(f"Label column '{label_col}' missing in training data")

    X = clean[feature_cols].copy()
    y = clean[label_col].copy()

    # coerce numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # drop rows with NaN / Inf in features or labels
    finite_mask = X.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1) & y.notnull() & np.isfinite(y)
    n_before = len(X)
    X = X.loc[finite_mask].reset_index(drop=True)
    y = y.loc[finite_mask].reset_index(drop=True)
    n_after = len(X)
    logger.info("train_xgb_confirm: rows before cleaning=%d, after=%d", n_before, n_after)

    if n_after == 0:
        raise ValueError("No training data left after cleaning (all rows dropped due to NaN/Inf).")

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Only one class present in labels: {unique_classes}. Need both classes to train.")

    # stratified split when possible
    stratify = y if len(unique_classes) == 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=stratify
    )
    logger.info("train_xgb_confirm: train rows=%d, val rows=%d", len(X_train), len(X_val))
    if len(X_train) < 2 or len(X_val) < 1:
        raise ValueError("Not enough data after train/val split. Try increasing dataset or changing test_size.")

    # --------------------- prepare DMatrix & params --------------------------
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "verbosity": 1 if verbose else 0,
    }

    evals = [(dval, "validation")]

    # --------------------- train via xgb.train (supports early stopping robustly) ----------------
    try:
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            evals=evals,
            early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds and int(early_stopping_rounds) > 0 else None,
            verbose_eval=verbose,
        )
    except TypeError as te:
        # fallback: some xgboost builds may not accept early_stopping_rounds param here;
        # try without early stopping and warn.
        logger.warning("xgb.train raised TypeError (%s). Retrying without early stopping.", te)
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            evals=evals,
            verbose_eval=verbose,
        )

    # wrap booster for sklearn-like interface
    model_wrapper = BoosterWrapper(bst, feature_names=feature_cols)

    # --------------------- evaluation metrics on validation set ----------------
    try:
        y_proba_val = model_wrapper.predict_proba(X_val)[:, 1]
        y_pred_val = (y_proba_val >= 0.5).astype(int)
    except Exception as exc:
        logger.warning("Could not compute proba on validation set via booster: %s. Falling back to zeros.", exc)
        y_proba_val = np.zeros(len(y_val), dtype=float)
        y_pred_val = np.zeros(len(y_val), dtype=int)

    metrics: Dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "accuracy": float(accuracy_score(y_val, y_pred_val)),
        "f1": float(f1_score(y_val, y_pred_val, zero_division=0)),
        "val_proba_mean": float(np.nanmean(y_proba_val)) if len(y_proba_val) > 0 else 0.0,
    }
    if len(np.unique(y_val)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba_val))
        except Exception:
            metrics["roc_auc"] = np.nan

    if verbose:
        logger.info("Training metrics: %s", metrics)

    return model_wrapper, feature_cols, metrics


def predict_confirm_prob(model: Any, candidates: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """
    Predict positive-class probability for each row in `candidates`.

    Accepts a BoosterWrapper (or any model with predict_proba) and returns a pd.Series
    indexed like candidates.index with name 'confirm_proba'.
    """
    if candidates is None or len(candidates) == 0:
        return pd.Series(dtype=float)

    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")

    missing = [c for c in feature_cols if c not in candidates.columns]
    if missing:
        logger.warning("predict_confirm_prob: missing feature columns in candidates, filling with 0: %s", missing)
        for m in missing:
            candidates[m] = 0.0

    X = candidates[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # If model has predict_proba, use it; otherwise attempt predict and convert to float
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)
            proba = np.asarray(preds, dtype=float)
    except Exception as exc:
        logger.error("predict_confirm_prob failed: %s", exc)
        proba = np.zeros(len(X), dtype=float)

    return pd.Series(proba, index=candidates.index, name="confirm_proba")
