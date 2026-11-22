"""
privacy_accountant.py
---------------------
Rényi-DP accounting for the **sampled Gaussian mechanism** (SGM), which is what
one step of DP-SGD is: Poisson-subsample the data with rate q, clip each
per-record gradient to L2 norm C, sum, and add N(0, (σC)² I).

Implements the integer-order RDP upper bound of Mironov, Talwar & Zhang,
"Rényi Differential Privacy of the Sampled Gaussian Mechanism" (2019),
Section 3.3:

    RDP(α) = (1 / (α−1)) · log( Σ_{k=0}^{α} C(α,k) (1−q)^{α−k} q^k
                                 · exp( k(k−1) / (2σ²) ) )

for integer α ≥ 2, composed linearly over steps, then converted to (ε, δ) by

    ε(δ) = min_α [ RDP(α) + log(1/δ) / (α−1) ].

This is the same computation performed by Opacus / TF-Privacy restricted to
integer orders (a valid upper bound; fractional orders tighten ε by only a few
percent). Self-contained (numpy only) and MIT-licensed, in keeping with the
programme's licence-clean rule.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_ORDERS = tuple(range(2, 129))


def _log_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of one SGM step at integer order alpha (nats)."""
    if q == 0.0:
        return 0.0
    if q > 1.0 or q < 0.0:
        raise ValueError("sampling rate q must be in [0, 1]")
    if q == 1.0:                       # plain Gaussian mechanism
        return alpha / (2.0 * sigma ** 2)
    # log-sum-exp over the binomial expansion
    terms = []
    for k in range(alpha + 1):
        t = (_log_comb(alpha, k)
             + (alpha - k) * math.log1p(-q)
             + k * math.log(q)
             + (k * (k - 1)) / (2.0 * sigma ** 2))
        terms.append(t)
    m = max(terms)
    s = sum(math.exp(t - m) for t in terms)
    return (m + math.log(s)) / (alpha - 1)


def compose_epsilon(q: float, sigma: float, steps: int, delta: float,
                    orders=DEFAULT_ORDERS) -> float:
    """(ε, δ)-DP guarantee after `steps` SGM steps at rate q, noise σ."""
    best = math.inf
    for a in orders:
        rdp = steps * rdp_sampled_gaussian(q, sigma, a)
        eps = rdp + math.log(1.0 / delta) / (a - 1)
        best = min(best, eps)
    return best


def calibrate_sigma(target_eps: float, q: float, steps: int, delta: float,
                    lo: float = 0.4, hi: float = 200.0, tol: float = 1e-3) -> float:
    """Smallest σ whose composed ε is ≤ target_eps (bisection).

    ε(σ) is monotone decreasing in σ, so plain bisection converges. Raises if
    even σ = hi cannot reach the target (caller should lower q or steps).
    """
    if compose_epsilon(q, hi, steps, delta) > target_eps:
        raise ValueError(f"target ε={target_eps} unreachable even at σ={hi}; "
                         "reduce steps or sampling rate")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if compose_epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def gaussian_mechanism_epsilon(sigma_over_sensitivity: float,
                               delta: float) -> float:
    """(ε, δ) of a single full-batch Gaussian mechanism (q = 1, one step)."""
    return compose_epsilon(1.0, sigma_over_sensitivity, 1, delta)
