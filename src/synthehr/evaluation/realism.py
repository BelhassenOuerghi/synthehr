"""
Realism of a synthetic table, measured against REAL rows the generator never
saw (memorisation cannot pass for realism).

* :func:`detection_auc`: a classifier two-sample test (Lopez-Paz & Oquab,
  2017). Gradient-boosted trees try to tell real from synthetic rows; 0.5
  means indistinguishable. Compare with the real-versus-real control, which
  is rarely exactly 0.5.
* :func:`precision_recall`: improved precision and recall (Kynkaanniemi et
  al., 2019). Caution, found in the research: an over-dispersed synthetic
  cloud has huge neighbourhood balls and scores recall near 1 with precision
  near 0. Read them together, or use density and coverage.
* :func:`density_coverage` (Naeem et al., 2020): balls around real rows only,
  so over-dispersion cannot inflate them.
* :func:`marginals`: per-column Wasserstein-1 (numeric, in real standard
  deviations) and total-variation distance (categorical).
* :func:`correlations`: error in the numeric Spearman correlation matrix, and
  in each feature's correlation with the target (what a model actually uses).
* :func:`label_rate`: the share of positives. Ranking metrics ignore a wrong
  label rate; thresholded predictions and parity-based fairness do not.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from synthehr.privacy.attacks import encode


def _codes(df, discrete, cats):
    X = df.copy()
    for c in discrete:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c]).codes
    return X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")


def integer_columns(real: pd.DataFrame, discrete: Sequence[str]) -> list:
    out = []
    for c in real.columns:
        if c in discrete:
            continue
        v = pd.to_numeric(real[c], errors="coerce").dropna().to_numpy()
        if len(v) and np.all(np.equal(np.mod(v, 1), 0)):
            out.append(c)
    return out


def round_integer_columns(syn: pd.DataFrame, real_train: pd.DataFrame,
                          discrete: Sequence[str]) -> pd.DataFrame:
    """Round columns that are integer-valued in the real data (a free
    post-processing step that removes a trivial detection cue)."""
    syn = syn.copy()
    for c in integer_columns(real_train, discrete):
        syn[c] = np.round(pd.to_numeric(syn[c], errors="coerce"))
    return syn


def detection_auc(real: pd.DataFrame, syn: pd.DataFrame,
                  discrete: Sequence[str], seed: int = 0, n: int = 10000
                  ) -> float:
    from xgboost import XGBClassifier
    rng = np.random.default_rng(seed)
    n = min(n, len(real), len(syn))
    r = real.iloc[rng.choice(len(real), n, replace=False)]
    s = syn.iloc[rng.choice(len(syn), n, replace=False)][real.columns]
    cats = {c: sorted(set(real[c].astype(str)) | set(syn[c].astype(str)))
            for c in discrete}
    X = np.concatenate([_codes(r, discrete, cats), _codes(s, discrete, cats)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True,
                                  random_state=seed).split(X, y):
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            subsample=0.9, n_jobs=4, random_state=seed,
                            eval_metric="logloss").fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def _sample_encoded(real_train, real, syn, discrete, n, seed):
    rng = np.random.default_rng(seed)
    n = min(n, len(real), len(syn))
    r = real.iloc[rng.choice(len(real), n, replace=False)]
    s = syn.iloc[rng.choice(len(syn), n, replace=False)][real.columns]
    _, Xr, Xs = encode(real_train, r, s, discrete=list(discrete))
    return Xr, Xs


def _knn_radius(X, k):
    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, k]


def precision_recall(real_train, real, syn, discrete, k: int = 5,
                     n: int = 5000, seed: int = 0) -> dict:
    Xr, Xs = _sample_encoded(real_train, real, syn, discrete, n, seed)

    def hits(ref, query):
        radius = _knn_radius(ref, k)
        hit = np.zeros(len(query), dtype=bool)
        for i in range(0, len(query), 1000):
            hit[i:i + 1000] = (pairwise_distances(query[i:i + 1000], ref)
                               <= radius[None, :]).any(axis=1)
        return float(hit.mean())

    return {"precision": hits(Xr, Xs), "recall": hits(Xs, Xr)}


def density_coverage(real_train, real, syn, discrete, k: int = 5,
                     n: int = 5000, seed: int = 0) -> dict:
    Xr, Xs = _sample_encoded(real_train, real, syn, discrete, n, seed)
    radius = _knn_radius(Xr, k)
    inside = np.zeros(len(Xs))
    covered = np.zeros(len(Xr), dtype=bool)
    for i in range(0, len(Xs), 1000):
        hit = pairwise_distances(Xs[i:i + 1000], Xr) <= radius[None, :]
        inside[i:i + 1000] = hit.sum(axis=1)
        covered |= hit.any(axis=0)
    return {"density": float(inside.sum() / (k * len(Xs))),
            "coverage": float(covered.mean())}


def marginals(real_train, real, syn, discrete) -> dict:
    w1, tvd = [], []
    for c in real.columns:
        if c in discrete:
            a = real[c].astype(str).value_counts(normalize=True)
            b = syn[c].astype(str).value_counts(normalize=True)
            cats = set(a.index) | set(b.index)
            tvd.append(0.5 * sum(abs(a.get(g, 0) - b.get(g, 0)) for g in cats))
        else:
            sd = pd.to_numeric(real_train[c], errors="coerce").std() or 1.0
            x = pd.to_numeric(real[c], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(syn[c], errors="coerce").dropna().to_numpy()
            if len(x) and len(y):
                w1.append(wasserstein_distance(x, y) / sd)
    return {"marginal_w1": float(np.mean(w1)) if w1 else None,
            "marginal_tvd": float(np.mean(tvd)) if tvd else None}


def correlations(real, syn, discrete, target: Optional[str]) -> dict:
    num = [c for c in real.columns if c not in discrete]
    rr = real[num].apply(pd.to_numeric, errors="coerce")
    ss = syn[num].apply(pd.to_numeric, errors="coerce")
    out = {}
    if len(num) >= 2:
        cr, cs = rr.corr("spearman").to_numpy(), ss.corr("spearman").to_numpy()
        iu = np.triu_indices(len(num), 1)
        out["corr_error"] = float(np.nanmean(np.abs(cr[iu] - cs[iu])))
    if target is not None:
        yr = pd.to_numeric(real[target], errors="coerce")
        ys = pd.to_numeric(syn[target], errors="coerce")
        feats = [c for c in num if c != target]
        if feats:
            ar = rr[feats].corrwith(yr, method="spearman").to_numpy()
            a_s = ss[feats].corrwith(ys, method="spearman").to_numpy()
            out["target_assoc_error"] = float(np.nanmean(np.abs(ar - a_s)))
    return out


def label_rate(df: pd.DataFrame, target: str, positive="1") -> float:
    return float((df[target].astype(str) == str(positive)).mean())


def realism(real_train: pd.DataFrame, real_test: pd.DataFrame,
            syn: pd.DataFrame, discrete: Sequence[str],
            target: Optional[str] = None, seed: int = 0) -> dict:
    """Every realism metric in one dictionary."""
    syn = syn[real_test.columns]
    out = {"detection_auc": detection_auc(real_test, syn, discrete, seed),
           "detection_auc_rounded": detection_auc(
               real_test, round_integer_columns(syn, real_train, discrete),
               discrete, seed)}
    out.update(precision_recall(real_train, real_test, syn, discrete,
                                seed=seed))
    out.update(density_coverage(real_train, real_test, syn, discrete,
                                seed=seed))
    out.update(marginals(real_train, real_test, syn, discrete))
    out.update(correlations(real_test, syn, discrete, target))
    return out
