"""
budget.py
---------
Privacy accounting for **pooled releases**: a synthetic dataset assembled from
K independently trained DP generators, all fitted on the same private rows.

Every run is a sequence of sampled Gaussian mechanism steps (plus, for
DP-CTGAN, one Gaussian release of the conditional category counts). Because
the runs all touch the same data, their Renyi-DP curves add (sequential
composition); the pooled (epsilon, delta) is then the usual RDP conversion.
No parallel-composition discount is claimed.

Two ensembling regimes, both measured:
  * "full":  each run at the release budget (epsilon = 10 on its own); the
             pooled release then spends the composed budget, which RDP makes
             sub-linear in K (four runs at 10 compose to about 23, not 40);
  * "split": a fixed total budget (epsilon = 10 for the pooled release), each
             run calibrated to a noise level such that the K runs compose to
             exactly the target. This is the decision-relevant comparison: at
             a fixed privacy cost, is one generator or a pool of K noisier
             generators the better release?
"""
from __future__ import annotations

import math

from privacy_accountant import DEFAULT_ORDERS, rdp_sampled_gaussian


def run(q, sigma, steps, sigma_hist=None):
    return {"q": q, "sigma": sigma, "steps": steps, "sigma_hist": sigma_hist}


def compose(runs, delta, orders=DEFAULT_ORDERS):
    """(epsilon, delta) of the composition of all `runs`."""
    best = math.inf
    for a in orders:
        rdp = 0.0
        for r in runs:
            rdp += r["steps"] * rdp_sampled_gaussian(r["q"], r["sigma"], a)
            if r.get("sigma_hist"):
                rdp += rdp_sampled_gaussian(1.0, r["sigma_hist"], a)
        best = min(best, rdp + math.log(1.0 / delta) / (a - 1))
    return best


def calibrate_split(target_eps, q, steps, k, delta, sigma_hist=None,
                    lo=0.4, hi=400.0, tol=1e-3):
    """Per-run noise multiplier such that k identical runs compose to
    `target_eps`."""
    def eps(sig):
        return compose([run(q, sig, steps, sigma_hist)] * k, delta)

    if eps(hi) > target_eps:
        raise ValueError("split budget unreachable; reduce steps or k")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if eps(mid) > target_eps else (lo, mid)
    return hi
