# model.py
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BoosterWrapper:
    """
    Wrap an xgboost.Booster to provide predict_proba / predict and utility methods.
    """
    def __init__(self, booster: xgb.Booster, feature_names: List[str]):
        self.booster = booster
        self.feature_names = feature_names
        self.best_iteration = getattr(booster, "best_iteration", None)

    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        Xp = X.reindex(columns=self.feature_names).fillna(0.0)
        return xgb.DMatrix(Xp, feature_names=self.feature_names)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        dmat = self._dmatrix(X)
        if self.best_iteration is not None:
            raw = self.booster.predict(dmat, iteration_range=(0, int(self.best_iteration)+1))
        else:
            raw = self.booster.predict(dmat)
        raw = np.asarray(raw, dtype=float)
        probs = np.vstack([1.0 - raw, raw]).T
        return pd.Series(probs[:, 1], index=X.index, name="confirm_proba")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def save_model(self, filepath: str):
        """Save the booster to disk using joblib."""
        joblib.dump({"booster": self.booster, "feature_names": self.feature_names}, filepath)
        logger.info("Model saved to %s", filepath)

    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved BoosterWrapper."""
        data = joblib.load(filepath)
        return cls(booster=data["booster"], feature_names=data["feature_names"])

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as DataFrame."""
        importance = self.booster.get_score(importance_type="weight")
        df = pd.DataFrame(
            [(k, importance.get(k, 0)) for k in self.feature_names],
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return df


def train_xgb_confirm(
    clean: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
    num_boost_round: int = 200,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[BoosterWrapper, List[str], Dict[str, Any]]:
    """Train XGBoost and return wrapper, features, and metrics."""
    # validation
    missing = [c for c in feature_cols if c not in clean.columns]
    if missing:
        raise KeyError(f"Missing feature columns in training data: {missing}")
    if label_col not in clean.columns:
        raise KeyError(f"Label column '{label_col}' missing in training data")

    X_all = clean[feature_cols].copy().apply(pd.to_numeric, errors="coerce")
    y_all = pd.to_numeric(clean[label_col], errors="coerce")

    mask_valid = X_all.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1) & y_all.notnull() & np.isfinite(y_all)
    X_all = X_all.loc[mask_valid].reset_index(drop=True)
    y_all = y_all.loc[mask_valid].reset_index(drop=True)

    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    logger.info("Labels: pos=%d neg=%d", n_pos, n_neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes present to train (pos={n_pos}, neg={n_neg})")

    scale_pos_weight = float(n_neg) / max(1.0, float(n_pos))
    stratify = y_all if len(np.unique(y_all)) == 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, shuffle=True, stratify=stratify
    )

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
        "scale_pos_weight": scale_pos_weight,
    }

    evals = [(dval, "validation")]
    try:
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            evals=evals,
            early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
            verbose_eval=verbose,
        )
    except TypeError as te:
        logger.warning("xgb.train TypeError (%s). Retrying without early_stopping_rounds.", te)
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            evals=evals,
            verbose_eval=verbose,
        )

    model_wrapper = BoosterWrapper(bst, feature_names=feature_cols)

    # evaluate
    y_proba_val = model_wrapper.predict_proba(X_val)
    y_pred_val = (y_proba_val >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "accuracy": float(accuracy_score(y_val, y_pred_val)),
        "f1": float(f1_score(y_val, y_pred_val, zero_division=0)),
        "val_proba_mean": float(np.nanmean(y_proba_val)) if y_proba_val.size > 0 else 0.0,
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
    """Predict positive-class probability for each row in candidates."""
    if candidates is None or len(candidates) == 0:
        return pd.Series(dtype=float)
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")

    missing = [c for c in feature_cols if c not in candidates.columns]
    if missing:
        logger.warning("predict_confirm_prob: missing feature cols, filling with 0: %s", missing)
        for m in missing:
            candidates[m] = 0.0

    X = candidates[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:
            preds = model.predict(X)
            proba = np.asarray(preds, dtype=float)
            proba = pd.Series(proba, index=candidates.index, name="confirm_proba")
    except Exception as exc:
        logger.error("predict_confirm_prob failed: %s", exc)
        proba = pd.Series(np.zeros(len(X), dtype=float), index=candidates.index, name="confirm_proba")

    return proba