"""
fairness_metrics.py
-------------------
Subgroup utility and group-fairness metrics for the downstream models, always
computed on the REAL test set (the deployment population), grouped by a
sensitive column that is available at test time.

Classification (readmission):
  * per-group ROC-AUC (groups with both classes and n >= min_n);
  * `auc_gap`, max−min subgroup AUC (performance disparity);
  * `dp_diff`, demographic-parity difference: max−min positive-
                           prediction rate at the 0.5 threshold;
  * `eo_diff`, equalized-odds difference: max over the pairwise
                           TPR gap and FPR gap (Hardt et al., 2016).

Regression (LOS):
  * per-group RMSE and mean signed error (systematic over/under-prediction);
  * `rmse_gap`, max−min subgroup RMSE;
  * `bias_gap`, max−min subgroup mean signed error.

Representation:
  * subgroup shares of a dataset vs the real training shares, the
    representation-bias statistic the fair-sampling arm exists to move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score


def _groups(sens, min_n):
    vc = sens.value_counts()
    return [g for g in vc.index if vc[g] >= min_n]


def classification_fairness(y_true, proba, sens, min_n=200):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    yhat = (p > 0.5).astype(int)
    sens = pd.Series(sens).astype(str).reset_index(drop=True)
    per, auc, tpr, fpr, pos = {}, {}, {}, {}, {}
    for g in _groups(sens, min_n):
        m = (sens == g).to_numpy()
        pos[g] = float(yhat[m].mean())
        if len(np.unique(y[m])) == 2:
            auc[g] = float(roc_auc_score(y[m], p[m]))
            tp = ((yhat == 1) & (y == 1) & m).sum(); fn = ((yhat == 0) & (y == 1) & m).sum()
            fp = ((yhat == 1) & (y == 0) & m).sum(); tn = ((yhat == 0) & (y == 0) & m).sum()
            tpr[g] = float(tp / max(tp + fn, 1)); fpr[g] = float(fp / max(fp + tn, 1))
        per[g] = {"n": int(m.sum()), "auc": auc.get(g), "pos_rate": pos[g],
                  "tpr": tpr.get(g), "fpr": fpr.get(g)}
    out = {"per_group": per}
    if len(auc) >= 2:
        out["auc_gap"] = float(max(auc.values()) - min(auc.values()))
    if len(pos) >= 2:
        out["dp_diff"] = float(max(pos.values()) - min(pos.values()))
    if len(tpr) >= 2:
        out["eo_diff"] = float(max(max(tpr.values()) - min(tpr.values()),
                                   max(fpr.values()) - min(fpr.values())))
    return out


def regression_fairness(y_true, pred, sens, min_n=200):
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(pred, dtype=float)
    sens = pd.Series(sens).astype(str).reset_index(drop=True)
    per, rmse, bias = {}, {}, {}
    for g in _groups(sens, min_n):
        m = (sens == g).to_numpy()
        rmse[g] = float(np.sqrt(mean_squared_error(y[m], f[m])))
        bias[g] = float(np.mean(f[m] - y[m]))
        per[g] = {"n": int(m.sum()), "rmse": rmse[g], "mean_error": bias[g]}
    out = {"per_group": per}
    if len(rmse) >= 2:
        out["rmse_gap"] = float(max(rmse.values()) - min(rmse.values()))
        out["bias_gap"] = float(max(bias.values()) - min(bias.values()))
    return out


def representation(df, real_train, cols):
    """Subgroup shares per sensitive column: dataset vs real training data."""
    out = {}
    for c in cols:
        syn_s = df[c].astype(str).value_counts(normalize=True)
        real_s = real_train[c].astype(str).value_counts(normalize=True)
        cats = sorted(set(syn_s.index) | set(real_s.index))
        out[c] = {g: {"real": float(real_s.get(g, 0.0)),
                      "dataset": float(syn_s.get(g, 0.0))} for g in cats}
    return out
