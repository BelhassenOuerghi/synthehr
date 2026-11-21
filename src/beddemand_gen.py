"""
beddemand_gen.py
----------------
Temporal counterpart to CTGAN for the bed-demand task.

CTGAN treats rows as exchangeable and therefore cannot synthesise a *single
autocorrelated count series*: shuffling the days would destroy exactly the
signal a forecaster needs. So for the temporal task the synthetic-data
generator is a **generative Poisson autoregressive model**, the same family of
count model used in the bed-demand replication, from which we sample a
synthetic trajectory of equal length. This keeps the study coherent (one
generative model per data modality) while being methodologically honest.

Downstream utility is measured the same way as the tabular tasks: a fixed
forecaster is trained on the *real* vs the *synthetic* history and both are
scored on the untouched real test span (TRTR vs TSTR).
"""
from __future__ import annotations

import numpy as np


def _design(y, W):
    T = len(y)
    X = np.ones((T - W, W + 1))
    for w in range(1, W + 1):
        X[:, w] = np.log1p(y[W - w:T - w])
    return X, y[W:]


def fit_poisson_ar(y, W=1):
    """Poisson AR(W) on log-counts by IRLS-free Newton on the Poisson NLL."""
    X, yt = _design(np.asarray(y, float), W)
    beta = np.zeros(X.shape[1]); beta[0] = np.log(max(yt.mean(), 0.5))
    for _ in range(50):
        mu = np.exp(np.clip(X @ beta, -10, 10))
        g = X.T @ (yt - mu)
        H = -(X.T * mu) @ X - 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def sample_series(beta, y_init, n, W=1, rng=None):
    """Sample a synthetic count trajectory of length n from a fitted Poisson-AR."""
    rng = rng or np.random.default_rng(0)
    buf = list(np.asarray(y_init, float)[-W:])
    out = []
    for _ in range(n):
        x = np.concatenate(([1.0], np.log1p(np.array(buf[::-1][:W]))))
        mu = float(np.exp(np.clip(x @ beta, -10, 10)))
        val = int(rng.poisson(mu))
        out.append(val); buf.append(val); buf = buf[-W:]
    return np.array(out, dtype=float)


def forecast(beta, history, horizon, W=1):
    """Expected-value multi-step forecast (plug-in mean), used for scoring."""
    buf = list(np.asarray(history, float)[-W:])
    out = []
    for _ in range(horizon):
        x = np.concatenate(([1.0], np.log1p(np.array(buf[::-1][:W]))))
        mu = float(np.exp(np.clip(x @ beta, -10, 10)))
        out.append(mu); buf.append(mu); buf = buf[-W:]
    return np.array(out)
