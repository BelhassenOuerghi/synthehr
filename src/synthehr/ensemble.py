"""
Pooled releases: one synthetic table assembled from several independently
trained private generators.

The research measured what pooling buys. With four private diffusion models
pooled, run-to-run spread in length-of-stay utility fell tenfold, but each
model spent its own budget, so the pool cost epsilon of about 23 rather than
10. Splitting a fixed budget four ways made each model so noisy that the pool
was worse than one model. Pooling trades privacy for predictability.

Every generator in a pool must have been trained on the same records; their
budgets compose sequentially (no parallel-composition discount).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from synthehr.privacy.accountant import compose


def pool(generators: Sequence, n: int, delta: Optional[float] = None,
         seed: int = 0):
    """Draw ``n`` rows split evenly across ``generators`` and shuffle them.

    Returns ``(release, epsilon)``; ``epsilon`` is the composed budget of all
    private generators (``None`` if any of them is not private)."""
    k = len(generators)
    per = [n // k + (1 if i < n % k else 0) for i in range(k)]
    frames = [g.sample(m) for g, m in zip(generators, per)]
    rng = np.random.default_rng(seed)
    out = pd.concat(frames, ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    runs = [getattr(g, "privacy_", None) for g in generators]
    if any(r is None for r in runs):
        return out, None
    delta = delta or generators[0].target_delta
    return out, compose(runs, delta)
