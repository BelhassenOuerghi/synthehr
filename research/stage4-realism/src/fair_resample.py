"""
fair_resample.py
----------------
Fair sampling for any generator, conditional or not: draw a large pool from
the (already private) generator, then resample it so that each sensitive
column's marginal matches a fair target. Weights are found by raking
(iterative proportional fitting, Deming & Stephan, 1940), which balances
several sensitive marginals jointly rather than one at a time.

Privacy: the pool, its subgroup shares and the target computed from them are
all functions of the DP generator's output, so the whole step is
post-processing and costs **no additional epsilon**, the same argument as
Stage 2's fair conditional sampling, made here without needing the
generator to accept a condition.

The target is Stage 2's: uniform over categories whose share is at least
`min_support`; rarer categories keep their share, so that a group the
generator has barely seen is not inflated into a large synthetic population.
"""
from __future__ import annotations

import numpy as np

from fair_dp_ctgan import fair_target


def rake_weights(pool, cols, targets, iters=100, tol=1e-6):
    w = np.ones(len(pool))
    keys = {c: pool[c].astype(str).to_numpy() for c in cols}
    for _ in range(iters):
        worst = 0.0
        for c in cols:
            cats, inv = np.unique(keys[c], return_inverse=True)
            cur = np.bincount(inv, weights=w) / w.sum()
            tgt = np.array([targets[c].get(g, 0.0) for g in cats])
            ratio = np.where(cur > 0, tgt / np.maximum(cur, 1e-12), 1.0)
            w = w * ratio[inv]
            worst = max(worst, float(np.abs(cur - tgt).max()))
        if worst < tol:
            break
    return w


def fair_resample(pool, n, cols, min_support=0.02, seed=0):
    targets = {}
    for c in cols:
        shares = pool[c].astype(str).value_counts(normalize=True).sort_index()
        t = fair_target(shares.to_numpy(), min_support)
        targets[c] = dict(zip(shares.index, t))
    w = rake_weights(pool, cols, targets)
    # Raking weights are constant within each joint cell of the sensitive
    # columns, so the release is drawn by exact stratified sampling: each
    # cell gets its raked share of n (largest-remainder rounding), drawn
    # uniformly within the cell. (Weighted sampling without replacement
    # would drift back towards the pool's own shares.)
    cell = pool[cols].astype(str).agg("|".join, axis=1).to_numpy()
    cells, inv = np.unique(cell, return_inverse=True)
    share = np.bincount(inv, weights=w) / w.sum()
    raw = share * n
    count = np.floor(raw).astype(int)
    count[np.argsort(raw - count)[::-1][:n - count.sum()]] += 1
    rng = np.random.default_rng(seed)
    idx, short = [], []
    for k in range(len(cells)):
        members = np.flatnonzero(inv == k)
        need = count[k]
        if need == 0:
            continue
        if need > len(members):          # cell too small: say so, top up
            short.append(cells[k])
            idx.append(rng.choice(members, need, replace=True))
        else:
            idx.append(rng.choice(members, need, replace=False))
    idx = rng.permutation(np.concatenate(idx))
    out = pool.iloc[idx].reset_index(drop=True)
    diag = {"targets": targets, "pool_rows": int(len(pool)),
            "pool_sufficient": not short, "short_cells": short,
            "duplicate_share": float(1 - len(np.unique(idx)) / n),
            "achieved": {c: out[c].astype(str).value_counts(normalize=True)
                         .to_dict() for c in cols}}
    return out, diag
