"""
Poisson autoregression for daily count series (admissions, bed occupancy).

A tabular generator treats rows as exchangeable, so it cannot synthesise a
single autocorrelated series: shuffling the days destroys the signal a
forecaster needs. This model is the temporal counterpart:

    y_t ~ Poisson(mu_t),  log mu_t = b0 + sum_w b_w * log(1 + y_{t-w})

fitted by Newton's method on the Poisson likelihood, or, with ``epsilon``,
by differentially private full-batch gradient ascent: each day's gradient is
clipped, and since one day appears in at most W + 1 rows (once as target, W
times as a lag) the clipped sum has sensitivity 2 (W + 1) C under
replace-one adjacency. Every trajectory sampled from the private
coefficients inherits the guarantee.

Numerically identical to the research implementation
(``research/common/beddemand_gen.py`` and ``dp_beddemand.py``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from synthehr.privacy.accountant import Run, calibrate_sigma, compose


def _design(y: np.ndarray, W: int):
    T = len(y)
    X = np.ones((T - W, W + 1))
    for w in range(1, W + 1):
        X[:, w] = np.log1p(y[W - w:T - w])
    return X, y[W:]


class PoissonAR:
    """Poisson AR(W) count-series generator.

    Parameters
    ----------
    window:
        Number of lags W.
    epsilon, delta:
        If ``epsilon`` is given, fit privately to that budget.
    clip_norm, iters, lr:
        DP fitting settings (ignored without ``epsilon``).
    """

    def __init__(self, window: int = 1, epsilon: Optional[float] = None,
                 delta: float = 1e-4, clip_norm: float = 1.0,
                 iters: int = 200, lr: float = 0.1, seed: int = 0):
        self.W = window
        self.target_epsilon, self.target_delta = epsilon, delta
        self.clip_norm, self.iters, self.lr = clip_norm, iters, lr
        self.seed = seed
        self.beta_: Optional[np.ndarray] = None
        self.epsilon_: Optional[float] = None
        self.privacy_: Optional[Run] = None

    def fit(self, y) -> "PoissonAR":
        y = np.asarray(y, float)
        if self.target_epsilon is None:
            self.beta_ = self._fit_mle(y)
        else:
            self.beta_ = self._fit_private(y)
        self.history_ = y[-self.W:].copy()
        return self

    def _fit_mle(self, y):
        X, yt = _design(y, self.W)
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

    def _fit_private(self, y):
        rng = np.random.default_rng(self.seed)
        X, yt = _design(y, self.W)
        n, d = X.shape
        sensitivity = 2.0 * (self.W + 1) * self.clip_norm
        sigma = calibrate_sigma(self.target_epsilon, 1.0, self.iters,
                                self.target_delta)
        beta = np.zeros(d)
        for _ in range(self.iters):
            mu = np.exp(np.clip(X @ beta, -10, 10))
            per_row = X * (yt - mu)[:, None]
            norms = np.linalg.norm(per_row, axis=1, keepdims=True)
            per_row = per_row * np.minimum(1.0, self.clip_norm / (norms + 1e-12))
            grad = per_row.sum(axis=0) + rng.normal(0.0, sigma * sensitivity,
                                                    size=d)
            beta = beta + self.lr * grad / n
        self.sigma_ = sigma
        self.privacy_ = Run(1.0, sigma, self.iters)
        self.epsilon_ = compose([self.privacy_], self.target_delta)
        return beta

    def sample(self, n: int, start=None, rng=None) -> np.ndarray:
        """A synthetic trajectory of length ``n``, started from ``start``
        (default: the last W observed values)."""
        rng = rng or np.random.default_rng(0)
        buf = list(np.asarray(self.history_ if start is None else start,
                              float)[-self.W:])
        out = []
        for _ in range(n):
            x = np.concatenate(([1.0], np.log1p(np.array(buf[::-1][:self.W]))))
            mu = float(np.exp(np.clip(x @ self.beta_, -10, 10)))
            val = int(rng.poisson(mu))
            out.append(val); buf.append(val); buf = buf[-self.W:]
        return np.array(out, dtype=float)

    def forecast(self, history, horizon: int) -> np.ndarray:
        """Plug-in mean forecast ``horizon`` steps ahead of ``history``."""
        buf = list(np.asarray(history, float)[-self.W:])
        out = []
        for _ in range(horizon):
            x = np.concatenate(([1.0], np.log1p(np.array(buf[::-1][:self.W]))))
            mu = float(np.exp(np.clip(x @ self.beta_, -10, 10)))
            out.append(mu); buf.append(mu); buf = buf[-self.W:]
        return np.array(out)
