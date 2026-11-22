"""
Membership-inference attacks against a synthetic release.

The attacker holds a candidate record and the synthetic data and must decide
whether the candidate was in the generator's training set. Members are real
training rows, non-members real rows the generator never saw (use a
patient-disjoint holdout: if one patient's stays sit on both sides, the
"non-members" are not unseen). AUC 0.5 is chance.

* **DCR attack:** score = minus the distance to the closest synthetic record;
  memorised rows sit closer to the synthetic cloud.
* **k-NN classifier attack:** distances to the 1st, 5th and 10th nearest
  synthetic records and their mean, fed to a logistic regression trained on
  half the candidates and scored on the other half.
* **Low false-positive regime** (Carlini et al., 2022): the DCR attack's
  true-positive rate at 1% and 0.1% false positives. An attack can be at
  chance on average and still single out a few records.

Distances are computed on standardised numerics (training statistics) and
one-hot categories. Numerically identical to the research attacks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def encode(reference: pd.DataFrame, *frames: pd.DataFrame,
           discrete: Sequence[str] = ()) -> list:
    """Encode ``frames`` with numerics z-scored by ``reference`` statistics
    and categories one-hot over ``reference`` categories. Returns the
    encoded reference followed by the encoded frames."""
    num = [c for c in reference.columns if c not in discrete]
    stats = pd.concat([pd.to_numeric(reference[c], errors="coerce")
                       for c in num], axis=1, keys=num)
    mu, sd = stats.mean(), stats.std().replace(0, 1.0)
    out = []
    for df in (reference,) + frames:
        parts = []
        for c in num:
            x = (pd.to_numeric(df[c], errors="coerce") - mu[c]) / sd[c]
            parts.append(x.fillna(0.0).to_numpy(dtype="float32").reshape(-1, 1))
        for c in discrete:
            cats = sorted(reference[c].astype(str).unique().tolist())
            oh = np.zeros((len(df), len(cats)), dtype="float32")
            idx = {v: i for i, v in enumerate(cats)}
            for i, v in enumerate(df[c].astype(str)):
                if v in idx:
                    oh[i, idx[v]] = 1.0
            parts.append(oh)
        out.append(np.concatenate(parts, axis=1))
    return out


def membership_inference(train: pd.DataFrame, holdout: pd.DataFrame,
                         synthetic: pd.DataFrame, discrete: Sequence[str],
                         n_candidates: int = 8000, seed: int = 42) -> dict:
    """Run both attacks; returns AUCs, the low-false-positive rates and the
    near-copy statistics of the synthetic data."""
    rng = np.random.default_rng(seed)
    n = min(n_candidates, len(train), len(holdout))
    tr_idx = rng.choice(len(train), n, replace=False)
    te_idx = rng.choice(len(holdout), n, replace=False)
    members, non_members = train.iloc[tr_idx], holdout.iloc[te_idx]
    Xtr, Xte, Xsy = encode(members, non_members, synthetic,
                           discrete=list(discrete))

    nn = NearestNeighbors(n_neighbors=10).fit(Xsy)
    d_mem, _ = nn.kneighbors(Xtr)
    d_non, _ = nn.kneighbors(Xte)
    y = np.concatenate([np.ones(n), np.zeros(n)])

    dcr = -np.concatenate([d_mem[:, 0], d_non[:, 0]])
    auc_dcr = float(roc_auc_score(y, dcr))
    fpr, tpr, _ = roc_curve(y, dcr)

    feats = np.concatenate([
        np.stack([d[:, 0], d[:, 4], d[:, 9], d.mean(axis=1)], axis=1)
        for d in (d_mem, d_non)])
    Xa, Xb, ya, yb = train_test_split(feats, y, test_size=0.5,
                                      random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xa, ya)
    auc_knn = float(roc_auc_score(yb, clf.predict_proba(Xb)[:, 1]))

    nn_tr = NearestNeighbors(n_neighbors=1).fit(Xtr)
    dcr_syn = nn_tr.kneighbors(Xsy[rng.choice(len(Xsy), min(len(Xsy), n),
                                              replace=False)])[0][:, 0]
    return {
        "attack_auc_dcr": auc_dcr,
        "attack_auc_knn_classifier": auc_knn,
        "tpr@fpr=0.01": float(np.interp(0.01, fpr, tpr)),
        "tpr@fpr=0.001": float(np.interp(0.001, fpr, tpr)),
        "syn_to_train_dcr_median": float(np.median(dcr_syn)),
        "syn_to_train_dcr_q05": float(np.quantile(dcr_syn, 0.05)),
        "near_duplicate_share@0.1": float((dcr_syn < 0.1).mean()),
        "n_candidates_per_class": int(n),
    }
