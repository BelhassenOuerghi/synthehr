"""
fidelity.py
-----------
Realism metrics for a synthetic table, all computed against **real rows the
generator never saw** (the real test split), so that memorisation cannot pass
for realism. Stages 0-2 measured realism only indirectly, through downstream
utility; Pillar 4 is about realism, so it measures it directly.

  * `detection`: a classifier two-sample test (Lopez-Paz & Oquab, 2017). A
    gradient-boosted classifier tries to tell real from synthetic rows;
    cross-validated ROC-AUC 0.5 means indistinguishable, 1.0 means trivially
    separable. Reported on the raw release and after rounding columns that are
    integer-valued in the real data (a free post-processing step), so that the
    reader can see how much of the gap is mere formatting.
  * `precision` / `recall`: improved precision and recall for generative
    models (Kynkaanniemi et al., 2019), with k-nearest-neighbour balls in a
    standardised, one-hot encoded space. Precision is the share of synthetic
    rows that land inside the real data's manifold (are they plausible?),
    recall the share of real rows covered by the synthetic manifold (is the
    diversity there?).
  * `marginal_w1` / `marginal_tvd`: mean per-column Wasserstein-1 distance
    (numeric, in units of the real standard deviation) and total-variation
    distance (discrete).
  * `corr_error`: mean absolute difference between the Spearman correlation
    matrices of the numeric columns.
  * `target_assoc_error`: mean absolute difference, over features, of each
    feature's Spearman correlation with the prediction target. This is the
    part of the joint distribution that a downstream model actually uses.

`mia_tail` complements the frozen harness's membership-inference AUC with the
true-positive rate at 1% and 0.1% false-positive rate (Carlini et al., 2022):
an attack can sit at chance on average and still single out a few records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from membership_inference import _encode


def _integer_columns(real, discrete):
    out = []
    for c in real.columns:
        if c in discrete:
            continue
        v = pd.to_numeric(real[c], errors="coerce").dropna().to_numpy()
        if len(v) and np.all(np.equal(np.mod(v, 1), 0)):
            out.append(c)
    return out


def round_integer_columns(syn, real_train, discrete):
    syn = syn.copy()
    for c in _integer_columns(real_train, discrete):
        syn[c] = np.round(pd.to_numeric(syn[c], errors="coerce"))
    return syn


def _codes(df, discrete, cats):
    X = df.copy()
    for c in discrete:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c]).codes
    return X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")


def detection_auc(real, syn, discrete, seed=0, n=10000):
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


def precision_recall(real_train, real, syn, discrete, k=5, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n, len(real), len(syn))
    r = real.iloc[rng.choice(len(real), n, replace=False)]
    s = syn.iloc[rng.choice(len(syn), n, replace=False)][real.columns]
    _, Xr, Xs = _encode(real_train, r, s, discrete)

    def manifold_hits(ref, query):
        # radius of each reference point's ball = distance to its k-th
        # neighbour (column 0 is the point itself); a query is inside the
        # manifold if it falls in ANY ball, checked exactly
        radius = NearestNeighbors(n_neighbors=k + 1).fit(ref).kneighbors(
            ref)[0][:, k]
        hit = np.zeros(len(query), dtype=bool)
        for i in range(0, len(query), 1000):
            dist = pairwise_distances(query[i:i + 1000], ref)
            hit[i:i + 1000] = (dist <= radius[None, :]).any(axis=1)
        return float(hit.mean())

    return {"precision": manifold_hits(Xr, Xs), "recall": manifold_hits(Xs, Xr)}


def marginals(real_train, real, syn, discrete):
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


def correlations(real, syn, discrete, target):
    num = [c for c in real.columns if c not in discrete]
    rr = real[num].apply(pd.to_numeric, errors="coerce")
    ss = syn[num].apply(pd.to_numeric, errors="coerce")
    cr, cs = rr.corr("spearman").to_numpy(), ss.corr("spearman").to_numpy()
    iu = np.triu_indices(len(num), 1)
    corr_err = float(np.nanmean(np.abs(cr[iu] - cs[iu])))
    yr = pd.to_numeric(real[target], errors="coerce")
    ys = pd.to_numeric(syn[target], errors="coerce")
    feats = [c for c in num if c != target]
    ar = rr[feats].corrwith(yr, method="spearman").to_numpy()
    as_ = ss[feats].corrwith(ys, method="spearman").to_numpy()
    return {"corr_error": corr_err,
            "target_assoc_error": float(np.nanmean(np.abs(ar - as_)))}


def fidelity(real_train, real_test, syn, discrete, target, seed=0):
    syn = syn[real_test.columns]
    out = {"detection_auc": detection_auc(real_test, syn, discrete, seed),
           "detection_auc_rounded": detection_auc(
               real_test, round_integer_columns(syn, real_train, discrete),
               discrete, seed)}
    out.update(precision_recall(real_train, real_test, syn, discrete,
                                seed=seed))
    out.update(marginals(real_train, real_test, syn, discrete))
    out.update(correlations(real_test, syn, discrete, target))
    return out


def mia_tail(train, test, syn, discrete, n_candidates=8000, seed=42):
    """DCR attack with the frozen harness's candidate draw, reporting the
    low-false-positive regime as well as the AUC."""
    rng = np.random.default_rng(seed)
    n = min(n_candidates, len(train), len(test))
    tr_idx = rng.choice(len(train), n, replace=False)
    te_idx = rng.choice(len(test), n, replace=False)
    Xtr, Xte, Xsy = _encode(train.iloc[tr_idx], test.iloc[te_idx], syn,
                            discrete)
    nn = NearestNeighbors(n_neighbors=1).fit(Xsy)
    s = -np.concatenate([nn.kneighbors(Xtr)[0][:, 0],
                         nn.kneighbors(Xte)[0][:, 0]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    fpr, tpr, _ = roc_curve(y, s)

    def tpr_at(x):
        return float(np.interp(x, fpr, tpr))

    return {"dcr_auc": float(roc_auc_score(y, s)),
            "tpr@fpr=0.01": tpr_at(0.01), "tpr@fpr=0.001": tpr_at(0.001)}
