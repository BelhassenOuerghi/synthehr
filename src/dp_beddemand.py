"""
dp_beddemand.py
---------------
Differentially-private counterpart of the Stage-0 Poisson-AR bed-demand
generator: the SAME model family (log-link Poisson autoregression, see
`beddemand_gen.py`), but fitted by **DP full-batch gradient ascent** so the
fitted coefficients, and every trajectory later sampled from them, carry an
(ε, δ) guarantee by post-processing.

Mechanism. Each day contributes one row (x_t = [1, log1p(y_{t-1}), ...],
target y_t) to the Poisson log-likelihood. Per-row gradients x_t·(y_t − μ_t)
are clipped to L2 norm `clip_norm`; one calendar day appears in at most
(W + 1) rows (once as target, W times as a lagged feature), so replace-one
adjacency moves the clipped gradient sum by at most 2·(W+1)·clip_norm. Each of
the `iters` full-batch steps is a Gaussian mechanism at that sensitivity
(q = 1), composed under the same RDP accountant as the tabular tasks; σ is
calibrated by bisection to meet the target ε.
"""
from __future__ import annotations

import numpy as np

from beddemand_gen import _design
from privacy_accountant import calibrate_sigma, compose_epsilon


def dp_fit_poisson_ar(y, W=1, *, target_epsilon=10.0, target_delta=1e-4,
                      clip_norm=1.0, iters=200, lr=0.1, seed=0, verbose=True):
    """DP Poisson-AR(W) fit; returns (beta, epsilon_spent, sigma)."""
    rng = np.random.default_rng(seed)
    X, yt = _design(np.asarray(y, float), W)
    n, d = X.shape
    sensitivity = 2.0 * (W + 1) * clip_norm

    sigma = calibrate_sigma(target_epsilon, 1.0, iters, target_delta)
    beta = np.zeros(d); beta[0] = 0.0        # data-independent start

    for it in range(iters):
        mu = np.exp(np.clip(X @ beta, -10, 10))
        per_row = X * (yt - mu)[:, None]                  # (n, d) row grads
        norms = np.linalg.norm(per_row, axis=1, keepdims=True)
        per_row = per_row * np.minimum(1.0, clip_norm / (norms + 1e-12))
        grad = per_row.sum(axis=0) + rng.normal(
            0.0, sigma * sensitivity, size=d)
        beta = beta + lr * grad / n
    eps = compose_epsilon(1.0, sigma, iters, target_delta)
    if verbose:
        print(f"  DP Poisson-AR: n={n} iters={iters} sigma={sigma:.3f} "
              f"eps={eps:.2f} beta={np.round(beta, 4).tolist()}", flush=True)
    return beta, eps, sigma
