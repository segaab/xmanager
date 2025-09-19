import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -----------------------------------------------------------------------------#
def train_xgb_confirm(
    clean: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
):
    """
    Trains an XGBoost model with sanity checks on data.

    Parameters
    ----------
    clean : pd.DataFrame
        Input dataframe containing features + labels.
    feature_cols : list[str]
        Columns to use as features.
    label_col : str
        Column name of target variable.
    test_size : float
        Fraction of data to reserve for validation.
    random_state : int
        Random seed.
    verbose : bool
        If True, print metrics and debug info.

    Returns
    -------
    model : xgb.XGBClassifier
    feature_cols : list[str]
    metrics : dict
    """

    # --- Data checks & sanitization ------------------------------------------
    missing = [c for c in feature_cols if c not in clean.columns]
    if missing:
        raise KeyError(f"Missing feature columns in training data: {missing}")
    if label_col not in clean.columns:
        raise KeyError(f"Label column '{label_col}' missing in training data")

    X = clean[feature_cols].copy()
    y = clean[label_col].copy()

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Drop NaN / inf rows
    mask = X.isnull().any(axis=1) | (~np.isfinite(X).all(axis=1)) | y.isnull() | (~np.isfinite(y))
    if mask.any():
        if verbose:
            print(f"train_xgb_confirm: dropping {mask.sum()} rows due to NaN/inf")
        X = X.loc[~mask].reset_index(drop=True)
        y = y.loc[~mask].reset_index(drop=True)

    if X.empty:
        raise ValueError("No training data left after cleaning")

    # --- Train/validation split ----------------------------------------------
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- Model ---------------------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0 if not verbose else 1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
    )

    # --- Metrics -------------------------------------------------------------
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_val)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba))
        except Exception:
            metrics["roc_auc"] = np.nan

    if verbose:
        print("Validation metrics:", metrics)

    return model, feature_cols, metrics