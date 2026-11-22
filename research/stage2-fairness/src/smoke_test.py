"""
smoke_test.py
-------------
Wiring check for the Stage-2 machinery on tiny FABRICATED data (no MIMIC
touched, single-core, under a minute):

  1. fair_target: uniformises eligible categories, leaves rare ones alone;
  2. FairDPCTGAN: fair-trains 2 epochs on a skewed 80/20 population, then
     sample_fair() must move the minority share decisively toward balance;
  3. fairness_metrics: classification + regression + representation wiring.

Run via `make smoke`. A wiring check, not a fidelity claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from fair_dp_ctgan import FairDPCTGAN, fair_target
from fairness_metrics import (classification_fairness, regression_fairness,
                              representation)


def main():
    rng = np.random.default_rng(0)

    # 1, fair_target math
    t = fair_target([0.60, 0.25, 0.12, 0.03], min_support=0.05)
    assert abs(t[:3].sum() + t[3] - 1) < 1e-9
    assert abs(t[0] - t[1]) < 1e-9 and abs(t[1] - t[2]) < 1e-9, t
    assert t[3] < 0.05, "rare category must not be inflated"
    print(f"fair_target ok: {np.round(t, 3).tolist()}")

    # 2, fair training + fair sampling on a skewed population
    n = 400
    df = pd.DataFrame({
        "age": rng.normal(65, 12, n).clip(18, 95),
        "lab": np.exp(rng.normal(0, 0.6, n)),
        "sex": rng.choice(["F", "M"], n, p=[0.2, 0.8]),   # skewed 80/20
    })
    m = FairDPCTGAN(fair_train_cols=["sex"], target_epsilon=8.0,
                    target_delta=1e-3, clip_norm=1.0, epochs=2,
                    batch_size=100, pac=5, vgm_components=3,
                    embedding_dim=16, generator_dim=(32,),
                    discriminator_dim=(32,), seed=0, verbose=False
                    ).fit(df, ["sex"])
    assert "sex" in m.noisy_shares_
    plan = m.build_plan(["sex"], min_support=0.05)
    syn = m.sample_fair(400, plan)
    share_f = (syn["sex"].astype(str) == "F").mean()
    assert list(syn.columns) == list(df.columns) and len(syn) == 400
    assert share_f > 0.30, f"fair sampling did not rebalance (F share {share_f:.2f})"
    print(f"fair_dp_ctgan ok: minority share 0.20 -> {share_f:.2f} "
          f"(target 0.50), eps spent {m.epsilon_spent_:.2f}")

    # 3, fairness metrics wiring
    y = rng.integers(0, 2, 600)
    p = np.clip(y * 0.6 + rng.normal(0.3, 0.2, 600), 0, 1)
    s = rng.choice(["A", "B"], 600)
    cf = classification_fairness(y, p, pd.Series(s), min_n=50)
    rf = regression_fairness(rng.normal(5, 2, 600), rng.normal(5, 2, 600),
                             pd.Series(s), min_n=50)
    rep = representation(syn, df, ["sex"])
    assert 0 <= cf["eo_diff"] <= 1 and rf["rmse_gap"] >= 0
    assert set(rep["sex"].keys()) == {"F", "M"}
    print(f"fairness_metrics ok: eo_diff={cf['eo_diff']:.3f}, "
          f"rmse_gap={rf['rmse_gap']:.3f}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
