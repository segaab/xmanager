# model.py
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_xgb_confirm(df: pd.DataFrame, feature_cols: list, label_col: str,
                      save_path=None, num_boost_round: int = 500,
                      early_stopping_rounds: int = 20, test_size: float = 0.2,
                      random_state: int = 42, verbose: bool = False):
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective':'multi:softprob',
        'num_class':3,
        'eval_metric':'mlogloss',
        'seed': random_state
    }

    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds, evals=evals, verbose_eval=verbose)

    y_pred = bst.predict(dtest)
    y_pred_labels = y_pred.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred_labels)
    metrics = {'accuracy': float(acc)}

    if save_path:
        bst.save_model(save_path)

    return bst, feature_cols, metrics

def predict_confirm_prob(model, df: pd.DataFrame, feature_cols: list) -> pd.Series:
    dmat = xgb.DMatrix(df[feature_cols])
    probs = model.predict(dmat)
    # return probability of label=1 (long confirmation)
    return pd.Series(probs[:,1], index=df.index)