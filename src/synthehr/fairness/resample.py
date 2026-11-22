"""
Fair sampling for any generator, conditional or not.

Draw a pool from a (private) generator, then resample it so that each
sensitive column's marginal matches a fair target. Weights come from raking
(iterative proportional fitting, Deming & Stephan, 1940), which balances
several sensitive columns jointly; the release is drawn by exact stratified
sampling within the raked cells (weighted sampling without replacement would
drift back towards the pool's own shares).

Every input is a function of the generator's output, so for a private
generator the step is post-processing and costs no additional epsilon.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def fair_target(shares, min_support: float = 0.02) -> np.ndarray:
    """Uniform over categories whose share is at least ``min_support``;
    rarer categories keep their share, so a group the data barely contains is
    not inflated into a large synthetic population."""
    p = np.asarray(shares, dtype=float)
    p = p / p.sum()
    eligible = p >= min_support
    if eligible.sum() <= 1:
        return p
    rare_mass = p[~eligible].sum()
    out = np.where(eligible, (1.0 - rare_mass) / eligible.sum(), p)
    return out / out.sum()


def rake_weights(pool: pd.DataFrame, columns: Sequence[str], targets: dict,
                 iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Row weights whose weighted marginals match ``targets[column]``."""
    w = np.ones(len(pool))
    keys = {c: pool[c].astype(str).to_numpy() for c in columns}
    for _ in range(iters):
        worst = 0.0
        for c in columns:
            cats, inv = np.unique(keys[c], return_inverse=True)
            cur = np.bincount(inv, weights=w) / w.sum()
            tgt = np.array([targets[c].get(g, 0.0) for g in cats])
            ratio = np.where(cur > 0, tgt / np.maximum(cur, 1e-12), 1.0)
            w = w * ratio[inv]
            worst = max(worst, float(np.abs(cur - tgt).max()))
        if worst < tol:
            break
    return w


def fair_resample(pool: pd.DataFrame, n: int, columns: Sequence[str],
                  min_support: float = 0.02, seed: int = 0):
    """Resample ``n`` rows from ``pool`` to fair marginals on ``columns``.

    Returns ``(release, diagnostics)``; diagnostics record the targets, the
    achieved shares, whether the pool was large enough, and the share of
    duplicated rows (non-zero only when a cell had to be topped up)."""
    targets = {}
    for c in columns:
        shares = pool[c].astype(str).value_counts(normalize=True).sort_index()
        t = fair_target(shares.to_numpy(), min_support)
        targets[c] = dict(zip(shares.index, t))
    w = rake_weights(pool, columns, targets)
    cell = pool[list(columns)].astype(str).agg("|".join, axis=1).to_numpy()
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
        if need > len(members):
            short.append(cells[k])
            idx.append(rng.choice(members, need, replace=True))
        else:
            idx.append(rng.choice(members, need, replace=False))
    idx = rng.permutation(np.concatenate(idx))
    out = pool.iloc[idx].reset_index(drop=True)
    diagnostics = {
        "targets": targets, "pool_rows": int(len(pool)),
        "pool_sufficient": not short, "short_cells": short,
        "duplicate_share": float(1 - len(np.unique(idx)) / n),
        "achieved": {c: out[c].astype(str).value_counts(normalize=True)
                     .to_dict() for c in columns}}
    return out, diagnostics
