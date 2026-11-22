"""
smoke_test.py
-------------
Minimal end-to-end sanity check of the Stage-1 machinery on a tiny FABRICATED
dataset (no MIMIC data touched, single-core, well under a minute):

  1. accountant: ε monotonicity + a spot value for the plain Gaussian mechanism;
  2. DPCTGAN: fit 2 epochs on 300 fabricated rows, sample, check schema + ε;
  3. DP Poisson-AR: fit on a fabricated count series, check finite β and ε;
  4. MIA: run both attacks on fabricated splits, check AUCs are probabilities.

Run via `make smoke`. This is a wiring check, not a fidelity claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from dp_beddemand import dp_fit_poisson_ar
from dp_ctgan import DPCTGAN
from membership_inference import run_mia
from privacy_accountant import compose_epsilon


def main():
    rng = np.random.default_rng(0)

    # 1, accountant
    e1 = compose_epsilon(0.01, 1.0, 1000, 1e-5)
    e2 = compose_epsilon(0.01, 2.0, 1000, 1e-5)
    assert e2 < e1, "more noise must mean less epsilon"
    assert 0 < e2 < e1 < 50, f"implausible epsilons {e1}, {e2}"
    print(f"accountant ok: eps(sigma=1)={e1:.2f} > eps(sigma=2)={e2:.2f}")

    # 2, DPCTGAN on fabricated mixed-type data
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(65, 12, n).clip(18, 95),
        "lab": np.exp(rng.normal(0, 0.6, n)),
        "sex": rng.choice(["F", "M"], n),
        "unit": rng.choice(["MICU", "SICU", "CCU"], n, p=[0.6, 0.3, 0.1]),
    })
    m = DPCTGAN(target_epsilon=8.0, target_delta=1e-3, clip_norm=1.0,
                epochs=2, batch_size=100, pac=5, vgm_components=3,
                embedding_dim=16, generator_dim=(32,), discriminator_dim=(32,),
                seed=0, verbose=False).fit(df, ["sex", "unit"])
    syn = m.sample(200)
    assert list(syn.columns) == list(df.columns)
    assert len(syn) == 200 and syn.notna().all().all()
    assert m.epsilon_spent_ is not None and m.epsilon_spent_ <= 8.0 + 1e-6
    print(f"dpctgan ok: sampled {len(syn)} rows, eps spent "
          f"{m.epsilon_spent_:.2f} <= 8.0")

    # 3, DP Poisson-AR on a fabricated count series
    y = rng.poisson(3.0, 400).astype(float)
    beta, eps, _ = dp_fit_poisson_ar(y, 1, target_epsilon=5.0,
                                     target_delta=1e-3, iters=50,
                                     seed=0, verbose=False)
    assert np.all(np.isfinite(beta)) and eps <= 5.0 + 1e-6
    print(f"dp poisson-ar ok: beta={np.round(beta, 3).tolist()}, "
          f"eps={eps:.2f} <= 5.0")

    # 4, MIA wiring
    tr = df.iloc[:150].reset_index(drop=True)
    te = df.iloc[150:].reset_index(drop=True)
    r = run_mia(tr, te, syn, ["sex", "unit"], n_candidates=100, seed=0)
    for k in ("attack_auc_dcr", "attack_auc_knn_classifier"):
        assert 0.0 <= r[k] <= 1.0
    print(f"mia ok: {r}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
