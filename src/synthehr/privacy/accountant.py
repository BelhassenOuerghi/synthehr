"""
Rényi differential privacy accounting for the sampled Gaussian mechanism.

One DP-SGD step is the sampled Gaussian mechanism (SGM): sample records with
probability q, clip each contribution to norm C, sum, add N(0, (sigma C)^2).
Its Rényi DP at integer order alpha is bounded by Mironov, Talwar & Zhang
(2019, Section 3.3)::

    RDP(alpha) = 1/(alpha-1) * log sum_{k=0..alpha} C(alpha,k) (1-q)^(alpha-k)
                 q^k exp(k(k-1) / (2 sigma^2))

RDP composes by addition; conversion to (epsilon, delta) is
``min_alpha RDP(alpha) + log(1/delta)/(alpha-1)``. Integer orders give a valid
upper bound (fractional orders tighten epsilon by a few percent).

A *run* is ``Run(q, sigma, steps, sigma_hist)``: ``steps`` SGM steps plus,
optionally, one full-data Gaussian release with noise multiplier
``sigma_hist`` (DP-CTGAN's noisy category counts). Several runs over the same
records compose sequentially; no parallel-composition discount is claimed.
Pure numpy, MIT-licensed, numerically identical to the research accountant.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

DEFAULT_ORDERS = tuple(range(2, 129))


@dataclass(frozen=True)
class Run:
    """Everything the accountant needs to know about one private training run."""
    q: float                         # sampling rate per step
    sigma: float                     # noise multiplier (std / sensitivity)
    steps: int                       # number of SGM steps
    sigma_hist: Optional[float] = None   # one extra full-data release, if any

    def to_dict(self) -> dict:
        return asdict(self)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP (in nats) of one SGM step at integer order ``alpha``."""
    if q == 0.0:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError("sampling rate q must be in [0, 1]")
    if q == 1.0:                                   # plain Gaussian mechanism
        return alpha / (2.0 * sigma ** 2)
    terms = [_log_comb(alpha, k) + (alpha - k) * math.log1p(-q)
             + k * math.log(q) + (k * (k - 1)) / (2.0 * sigma ** 2)
             for k in range(alpha + 1)]
    m = max(terms)
    return (m + math.log(sum(math.exp(t - m) for t in terms))) / (alpha - 1)


def compose(runs: Iterable[Run], delta: float,
            orders: Iterable[int] = DEFAULT_ORDERS) -> float:
    """(epsilon, delta)-DP of the sequential composition of ``runs``."""
    runs = list(runs)
    best = math.inf
    for a in orders:
        rdp = 0.0
        for r in runs:
            rdp += r.steps * rdp_sampled_gaussian(r.q, r.sigma, a)
            if r.sigma_hist:
                rdp += rdp_sampled_gaussian(1.0, r.sigma_hist, a)
        best = min(best, rdp + math.log(1.0 / delta) / (a - 1))
    return best


def epsilon(q: float, sigma: float, steps: int, delta: float,
            sigma_hist: Optional[float] = None) -> float:
    """Epsilon of a single run."""
    return compose([Run(q, sigma, steps, sigma_hist)], delta)


def calibrate_sigma(target_epsilon: float, q: float, steps: int, delta: float,
                    sigma_hist: Optional[float] = None, runs: int = 1,
                    lo: float = 0.4, hi: Optional[float] = None,
                    tol: float = 1e-3) -> float:
    """Smallest noise multiplier (to ``tol``) such that ``runs`` identical
    runs compose to at most ``target_epsilon``. Bisection: epsilon is
    monotone decreasing in sigma. The default upper bracket (200 for one run,
    400 for several) is the one the research used, so calibrated values are
    reproduced exactly."""
    if hi is None:
        hi = 200.0 if runs == 1 else 400.0
    def eps(s: float) -> float:
        return compose([Run(q, s, steps, sigma_hist)] * runs, delta)

    if eps(hi) > target_epsilon:
        raise ValueError(f"epsilon={target_epsilon} unreachable even at "
                         f"sigma={hi}; reduce the number of steps or the "
                         "sampling rate")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if eps(mid) > target_epsilon else (lo, mid)
    return hi
