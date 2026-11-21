"""Tests for fair resampling and pooled-release accounting (fabricated data)."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import budget  # noqa: E402
from fair_resample import fair_resample  # noqa: E402
from privacy_accountant import compose_epsilon  # noqa: E402


def _pool(n=50000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(["F", "M"], n, p=[0.4, 0.6])
    ins = rng.choice(["Medicare", "Private", "Medicaid", "Other", "None"], n,
                     p=[0.55, 0.27, 0.10, 0.075, 0.005])
    return pd.DataFrame({"gender": g, "insurance": ins,
                         "x": rng.standard_normal(n)})


def test_fair_resample_hits_joint_marginal_targets():
    pool = _pool()
    out, diag = fair_resample(pool, 10000, ["gender", "insurance"], 0.02)
    a = diag["achieved"]
    assert abs(a["gender"]["F"] - 0.5) < 2e-3
    # four eligible insurance groups share (1 - rare mass) equally
    rare = diag["targets"]["insurance"]["None"]
    for g in ("Medicare", "Private", "Medicaid", "Other"):
        assert abs(a["insurance"][g] - (1 - rare) / 4) < 2e-3
    assert abs(a["insurance"].get("None", 0) - rare) < 2e-3
    assert len(out) == 10000


def test_fair_resample_flags_insufficient_pool():
    pool = _pool(n=4000)
    _, diag = fair_resample(pool, 3900, ["gender", "insurance"], 0.02)
    assert not diag["pool_sufficient"] and diag["duplicate_share"] > 0


def test_single_run_composition_matches_accountant():
    q, sig, steps, d = 0.03, 1.2, 3000, 1e-5
    assert math.isclose(budget.compose([budget.run(q, sig, steps)], d),
                        compose_epsilon(q, sig, steps, d), rel_tol=1e-12)


def test_split_calibration_composes_to_target():
    q, steps, d = 500 / 30000, 6000, 1e-5
    s = budget.calibrate_split(10.0, q, steps, 4, d, sigma_hist=32.0)
    eps = budget.compose([budget.run(q, s, steps, 32.0)] * 4, d)
    assert 9.98 < eps <= 10.0


def test_full_pool_is_sublinear():
    q, steps, d = 1024 / 30000, 2930, 1e-5
    from privacy_accountant import calibrate_sigma
    s = calibrate_sigma(10.0, q, steps, d)
    eps4 = budget.compose([budget.run(q, s, steps)] * 4, d)
    assert 10 < eps4 < 40


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS  {name}", flush=True)
            except Exception:
                import traceback
                fails += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{fails} failure(s)"); sys.exit(1 if fails else 0)
