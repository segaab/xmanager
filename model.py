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


def train_xgb_confirm(
    clean: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
    num_boost_round: int = 200,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[xgb.XGBClassifier, List[str], Dict[str, Any]]:
    """
    Train an XGBoost binary classifier for confirm/probability.

    Returns:
        (model, feature_cols, metrics)
    """
    # basic validation
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

    # drop rows with NaN / Inf
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

    # model construction
    model = xgb.XGBClassifier(
        n_estimators=int(num_boost_round),
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(random_state),
        n_jobs=-1,
        use_label_encoder=False,
        verbosity=1 if verbose else 0,
    )

    fit_kwargs = {}
    eval_set = [(X_val, y_val)]
    if early_stopping_rounds is not None and int(early_stopping_rounds) > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        fit_kwargs["eval_set"] = eval_set
    else:
        if verbose:
            fit_kwargs["eval_set"] = eval_set

    model.fit(X_train, y_train, **fit_kwargs)

    # evaluation
    y_pred = model.predict(X_val)
    metrics: Dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba))
            metrics["val_proba_mean"] = float(np.nanmean(y_proba))
    except Exception as exc:
        logger.warning("Could not compute probability metrics: %s", exc)
        metrics["roc_auc"] = np.nan

    if verbose:
        logger.info("Training metrics: %s", metrics)

    return model, feature_cols, metrics


def predict_confirm_prob(model: Any, candidates: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """
    Predict positive-class probability for each row in `candidates`.

    Returns a pd.Series indexed like `candidates.index`.
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