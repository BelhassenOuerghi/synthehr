"""
Utility: train on synthetic, test on real (TSTR), against the train-on-real
ceiling (TRTR).

A fixed downstream model is trained on the release and scored on real held-out
rows; only the training data changes between arms, so a difference in score
is attributable to the data. Defaults are the research harness's models:
gradient-boosted trees for binary classification (class-weighted by the
training set's own label rate) and a random forest for regression.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from synthehr.fairness.metrics import (classification_fairness,
                                       regression_fairness)


def infer_task(y: pd.Series) -> str:
    vals = set(y.astype(str).unique())
    return "classification" if len(vals) <= 2 else "regression"


def _binary(y: pd.Series, positive) -> np.ndarray:
    return (y.astype(str) == str(positive)).astype(int).to_numpy()


def _prep(discrete, numeric):
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]),
         discrete)])


def tstr(train: pd.DataFrame, test: pd.DataFrame, target: str,
         discrete_columns: Sequence[str], task: str = "auto",
         positive="1", sensitive: Sequence[str] = (), seed: int = 0,
         clip: Optional[tuple] = None, model=None,
         min_group_size: int = 200) -> dict:
    """Fit the downstream model on ``train`` and score it on ``test``.

    ``model`` may be any scikit-learn regressor or classifier (it is wrapped
    in the same preprocessing); ``clip`` bounds regression targets and
    predictions, as the research did for length of stay."""
    task = infer_task(test[target]) if task == "auto" else task
    feats = [c for c in train.columns if c != target]
    disc = [c for c in discrete_columns if c != target and c in feats]
    num = [c for c in feats if c not in disc]
    Xtr, Xte = train[feats], test[feats]

    if task == "classification":
        from xgboost import XGBClassifier
        ytr, yte = _binary(train[target], positive), _binary(test[target],
                                                             positive)
        pos, neg = max(ytr.sum(), 1), max((1 - ytr).sum(), 1)
        est = model or XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.9,
            eval_metric="logloss", scale_pos_weight=neg / pos, n_jobs=-1,
            random_state=seed)
        m = Pipeline([("p", _prep(disc, num)), ("m", est)]).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        out = {"roc_auc": float(roc_auc_score(yte, p)),
               "pr_auc": float(average_precision_score(yte, p)),
               "accuracy": float(accuracy_score(yte, (p > 0.5).astype(int)))}
        out["fairness"] = {c: classification_fairness(yte, p, test[c],
                                                      min_group_size)
                           for c in sensitive}
        return out

    ytr = pd.to_numeric(train[target], errors="coerce")
    yte = pd.to_numeric(test[target], errors="coerce")
    if clip is not None:
        ytr, yte = ytr.clip(*clip), yte.clip(*clip)
    est = model or RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                         n_jobs=-1, random_state=seed)
    m = Pipeline([("p", _prep(disc, num)), ("m", est)]).fit(Xtr, ytr)
    pred = m.predict(Xte)
    if clip is not None:
        pred = np.clip(pred, *clip)
    out = {"r2": float(r2_score(yte, pred)),
           "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
           "mae": float(mean_absolute_error(yte, pred))}
    out["fairness"] = {c: regression_fairness(yte, pred, test[c],
                                              min_group_size)
                       for c in sensitive}
    return out
