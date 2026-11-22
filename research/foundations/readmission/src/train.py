"""Model training: preprocessing, hyperparameter tuning and 5-fold CV.

Primary model: XGBoost binary classifier. Class imbalance (readmission is the
minority class, ~14%) is handled with `scale_pos_weight = n_neg / n_pos`,
recomputed per training fold. Hyperparameters are lightly tuned with a staged
grid search (mirroring the upstream Blinder pipeline), then the tuned model is
evaluated with 5-fold stratified cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, average_precision_score
from xgboost import XGBClassifier

ID_COLS = ["stay_id", "subject_id", "hadm_id"]
CATEGORICAL = ["gender", "admission_type", "admission_location",
               "insurance", "first_careunit"]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """One-hot encode categoricals, keep numeric NaNs (XGBoost handles them)."""
    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=ID_COLS + ["label"])
    cats = [c for c in CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cats, dummy_na=False)
    # Clean feature names for XGBoost (no []<> characters).
    X.columns = [c.replace("[", "(").replace("]", ")").replace("<", "lt")
                 for c in X.columns]
    return X, y, list(X.columns)


def _make_clf(cfg: dict, scale_pos_weight: float, **overrides) -> XGBClassifier:
    params = dict(cfg["model"]["base_params"])
    params.update({
        "scale_pos_weight": scale_pos_weight,
        "random_state": cfg["model"]["random_state"],
    })
    params.update(overrides)
    return XGBClassifier(**params)


def tune(X: pd.DataFrame, y: np.ndarray, cfg: dict) -> tuple[dict, float]:
    """Light grid search over the configured parameter grid (roc_auc scored)."""
    tuning = cfg["model"]["tuning"]
    if not tuning.get("enabled", True):
        return {}, float("nan")
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    base = _make_clf(cfg, spw)
    gs = GridSearchCV(
        estimator=base,
        param_grid=tuning["param_grid"],
        scoring=tuning["scoring"],
        cv=tuning["cv"],
        n_jobs=1,          # XGBoost already uses n_jobs threads internally
        verbose=0,
    )
    gs.fit(X, y)
    return gs.best_params_, float(gs.best_score_)


def cross_validate(X: pd.DataFrame, y: np.ndarray, cfg: dict,
                   best_params: dict) -> dict:
    """Stratified 5-fold CV; returns per-fold metrics, ROC-curve data and OOF probs."""
    n_splits = cfg["model"]["n_splits"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=cfg["model"]["random_state"])

    mean_fpr = np.linspace(0, 1, 100)
    tprs, fold_aucs, fold_ap = [], [], []
    oof_prob = np.zeros(len(y), dtype=float)

    for k, (tr, te) in enumerate(skf.split(X, y)):
        ytr = y[tr]
        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        clf = _make_clf(cfg, spw, **best_params)
        clf.fit(X.iloc[tr], ytr)
        prob = clf.predict_proba(X.iloc[te])[:, 1]
        oof_prob[te] = prob

        fpr, tpr, _ = roc_curve(y[te], prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        fold_aucs.append(auc(fpr, tpr))
        fold_ap.append(average_precision_score(y[te], prob))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return {
        "mean_fpr": mean_fpr,
        "tprs": np.array(tprs),
        "mean_tpr": mean_tpr,
        "fold_aucs": np.array(fold_aucs),
        "fold_ap": np.array(fold_ap),
        "oof_prob": oof_prob,
        "mean_auc": float(auc(mean_fpr, mean_tpr)),
    }


def final_fit(X: pd.DataFrame, y: np.ndarray, cfg: dict,
              best_params: dict) -> tuple[XGBClassifier, pd.Series]:
    """Fit on all data for feature-importance reporting (gain)."""
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    clf = _make_clf(cfg, spw, **best_params)
    clf.fit(X, y)
    booster = clf.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.Series(gain).reindex(X.columns).fillna(0.0).sort_values(ascending=False)
    return clf, imp
