"""
Subgroup performance and group-fairness metrics, always on REAL test data.

Classification: per-group ROC-AUC, the AUC gap, the demographic-parity
difference and the equalised-odds difference (Hardt et al., 2016) at a
threshold, plus a degeneracy flag: a classifier that predicts one class is
trivially "fair", so parity metrics are only meaningful when the positive
prediction rate is inside (2%, 98%) in every group.

Regression: per-group RMSE and mean signed error, and their gaps.
Representation: subgroup shares of a dataset against the real data.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score


def _groups(sens: pd.Series, min_n: int) -> list:
    vc = sens.value_counts()
    return [g for g in vc.index if vc[g] >= min_n]


def classification_fairness(y_true, proba, sensitive, min_n: int = 200,
                            threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    yhat = (p > threshold).astype(int)
    sens = pd.Series(sensitive).astype(str).reset_index(drop=True)
    per, auc, tpr, fpr, pos = {}, {}, {}, {}, {}
    for g in _groups(sens, min_n):
        m = (sens == g).to_numpy()
        pos[g] = float(yhat[m].mean())
        if len(np.unique(y[m])) == 2:
            auc[g] = float(roc_auc_score(y[m], p[m]))
            tp = ((yhat == 1) & (y == 1) & m).sum()
            fn = ((yhat == 0) & (y == 1) & m).sum()
            fp = ((yhat == 1) & (y == 0) & m).sum()
            tn = ((yhat == 0) & (y == 0) & m).sum()
            tpr[g] = float(tp / max(tp + fn, 1))
            fpr[g] = float(fp / max(fp + tn, 1))
        per[g] = {"n": int(m.sum()), "auc": auc.get(g), "pos_rate": pos[g],
                  "tpr": tpr.get(g), "fpr": fpr.get(g)}
    out = {"per_group": per,
           "degenerate": any(not 0.02 <= r <= 0.98 for r in pos.values())}
    if len(auc) >= 2:
        out["auc_gap"] = float(max(auc.values()) - min(auc.values()))
    if len(pos) >= 2:
        out["dp_diff"] = float(max(pos.values()) - min(pos.values()))
    if len(tpr) >= 2:
        out["eo_diff"] = float(max(max(tpr.values()) - min(tpr.values()),
                                   max(fpr.values()) - min(fpr.values())))
    return out


def regression_fairness(y_true, pred, sensitive, min_n: int = 200) -> dict:
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(pred, dtype=float)
    sens = pd.Series(sensitive).astype(str).reset_index(drop=True)
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


def representation(df: pd.DataFrame, real: pd.DataFrame,
                   columns: Sequence[str]) -> dict:
    """Subgroup shares per column: dataset against real data."""
    out = {}
    for c in columns:
        syn_s = df[c].astype(str).value_counts(normalize=True)
        real_s = real[c].astype(str).value_counts(normalize=True)
        cats = sorted(set(syn_s.index) | set(real_s.index))
        out[c] = {g: {"real": float(real_s.get(g, 0.0)),
                      "dataset": float(syn_s.get(g, 0.0))} for g in cats}
    return out
