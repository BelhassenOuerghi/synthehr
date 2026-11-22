"""
membership_inference.py
-----------------------
Empirical privacy evaluation: **membership-inference attacks (MIA) against the
synthetic data release**. The attacker holds a candidate record and the
synthetic dataset, and must decide whether the candidate was in the generator's
training set. Ground truth: real *training* rows are members, real *test* rows
(never seen by any generator) are non-members. An attack AUC of 0.5 is chance,
the release carries no membership signal; the gap above 0.5 is the leakage.

Two standard black-box attacks:

1. **DCR attack** (distance to closest record): score a candidate by the
   negative distance to its nearest synthetic neighbour. Memorised or
   near-copied training rows sit closer to the synthetic cloud than unseen
   rows. Also reports the DCR distribution of the synthetic data itself vs the
   training set (near-duplicate share), the direct copy-detection statistic.

2. **k-NN classifier attack**: features = distances to the 1st, 5th, 10th
   nearest synthetic neighbours + their mean; a logistic-regression attacker is
   trained on half the candidates and scored on the other half, a strictly
   stronger use of the same black-box signal.

Both operate on the model-ready representation (standardised continuous
columns, one-hot discrete) so distances are meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def _encode(train, test, syn, discrete):
    """Shared encoding: z-scored numeric (train stats), one-hot discrete."""
    frames = {"tr": train.copy(), "te": test.copy(), "sy": syn.copy()}
    num = [c for c in train.columns if c not in discrete]
    mu = pd.concat([pd.to_numeric(train[c], errors="coerce") for c in num],
                   axis=1, keys=num).mean()
    sd = pd.concat([pd.to_numeric(train[c], errors="coerce") for c in num],
                   axis=1, keys=num).std().replace(0, 1.0)
    out = {}
    for k, df in frames.items():
        parts = []
        for c in num:
            x = (pd.to_numeric(df[c], errors="coerce") - mu[c]) / sd[c]
            parts.append(x.fillna(0.0).to_numpy(dtype="float32").reshape(-1, 1))
        for c in discrete:
            cats = sorted(train[c].astype(str).unique().tolist())
            oh = np.zeros((len(df), len(cats)), dtype="float32")
            idx = {v: i for i, v in enumerate(cats)}
            for i, v in enumerate(df[c].astype(str)):
                if v in idx:
                    oh[i, idx[v]] = 1.0
            parts.append(oh)
        out[k] = np.concatenate(parts, axis=1)
    return out["tr"], out["te"], out["sy"]


def run_mia(train, test, syn, discrete, *, n_candidates=8000, seed=42):
    """Both attacks; returns a dict of attack AUCs + copy statistics."""
    rng = np.random.default_rng(seed)
    n = min(n_candidates, len(train), len(test))
    tr_idx = rng.choice(len(train), n, replace=False)
    te_idx = rng.choice(len(test), n, replace=False)
    Xtr, Xte, Xsy = _encode(train.iloc[tr_idx], test.iloc[te_idx], syn,
                            discrete)

    nn = NearestNeighbors(n_neighbors=10).fit(Xsy)
    d_mem, _ = nn.kneighbors(Xtr)          # (n, 10) distances, members
    d_non, _ = nn.kneighbors(Xte)          # non-members
    y = np.concatenate([np.ones(n), np.zeros(n)])

    # -- attack 1: DCR score = -(distance to closest synthetic record) -------
    scores = -np.concatenate([d_mem[:, 0], d_non[:, 0]])
    auc_dcr = float(roc_auc_score(y, scores))

    # -- attack 2: k-NN feature classifier -----------------------------------
    feats = np.concatenate([
        np.stack([d[:, 0], d[:, 4], d[:, 9], d.mean(axis=1)], axis=1)
        for d in (d_mem, d_non)])
    Xa, Xb, ya, yb = train_test_split(feats, y, test_size=0.5,
                                      random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xa, ya)
    auc_knn = float(roc_auc_score(yb, clf.predict_proba(Xb)[:, 1]))

    # -- copy detection: how close does the synthetic data sit to train? -----
    nn_tr = NearestNeighbors(n_neighbors=1).fit(Xtr)
    dcr_syn = nn_tr.kneighbors(Xsy[rng.choice(len(Xsy),
                                              min(len(Xsy), n), replace=False)]
                               )[0][:, 0]
    return {
        "attack_auc_dcr": auc_dcr,
        "attack_auc_knn_classifier": auc_knn,
        "syn_to_train_dcr_median": float(np.median(dcr_syn)),
        "syn_to_train_dcr_q05": float(np.quantile(dcr_syn, 0.05)),
        "near_duplicate_share@0.1": float((dcr_syn < 0.1).mean()),
        "n_candidates_per_class": int(n),
    }
