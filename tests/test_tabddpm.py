"""
Correctness tests for the from-scratch TabDDPM and its DP-SGD trainer.

All data here is fabricated; nothing touches MIMIC. Run with
    python -m pytest tests -q        (or: make test)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tabddpm import DPTabDDPM, TabDDPM, _seg_log_softmax  # noqa: E402


def _toy(n=3000, seed=0):
    """Two clusters: category 'a' -> x ~ N(-2, .3), 'b' -> x ~ N(+2, .3)."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b"], size=n, p=[0.7, 0.3])
    x = np.where(g == "a", -2.0, 2.0) + 0.3 * rng.standard_normal(n)
    y = x * 0.5 + 0.1 * rng.standard_normal(n)
    h = rng.choice(["u", "v", "w"], size=n)
    return pd.DataFrame({"x": x, "g": g, "y": y, "h": h})


def _dp(unit="stay", clip=1.0, m=3, **kw):
    df = _toy(400)
    model = DPTabDDPM(hidden=(32, 32), emb=16, timesteps=50, clip_norm=clip,
                      noise_multiplicity=m, unit=unit, chunk=64,
                      expected_batch=64, epochs=1, seed=1, verbose=False, **kw)
    model._setup(df, ["g", "h"])
    return model, df


def test_per_row_gradients_match_explicit_loop():
    model, _ = _dp()
    rows = np.arange(10)
    p = model._noised(rows, reps=model.m)
    g = model._per_row_grads(rows, p=p)
    for i in rows:
        sl = slice(i * model.m, (i + 1) * model.m)
        pi = {k: v[sl] for k, v in p.items()}
        model.model.zero_grad()
        loss = model._loss(model.model(pi["xin"], pi["t"]), pi)
        loss.backward()
        for k, prm in model.model.named_parameters():
            assert torch.allclose(g[k][i], prm.grad, atol=1e-5, rtol=1e-4), k


def test_clipped_sum_without_clipping_is_the_plain_gradient():
    model, _ = _dp(clip=1e9)
    rows = np.arange(37)
    model.chunk = len(rows)       # one unpadded chunk, so the replay below
    torch.manual_seed(0)          # draws exactly the same noise
    gen_state = model.gen.get_state()
    total = model._clipped_sum(rows, rows.copy())
    model.gen.set_state(gen_state)        # replay the same noise draws
    p = model._noised(rows, reps=model.m)
    model.model.zero_grad()
    out = model.model(p["xin"], p["t"])
    # sum over rows of each row's multiplicity-mean loss
    per = []
    for i in range(len(rows)):
        sl = slice(i * model.m, (i + 1) * model.m)
        per.append(model._loss(out[sl], {k: v[sl] for k, v in p.items()}))
    torch.stack(per).sum().backward()
    for k, prm in model.model.named_parameters():
        assert torch.allclose(total[k], prm.grad, atol=1e-4, rtol=1e-3), k


def test_patient_level_contributions_are_bounded():
    model, _ = _dp(unit="patient", clip=0.05)
    rows = np.arange(60)
    patients = rows // 4                      # 15 patients x 4 stays
    # each patient's clipped contribution alone must have norm <= C
    for pid in range(15):
        sel = rows[patients == pid]
        tot = model._clipped_sum(sel, patients[sel])
        norm = torch.sqrt(sum((v ** 2).sum() for v in tot.values()))
        assert float(norm) <= 0.05 * (1 + 1e-4)


def test_patient_chunks_never_split_a_patient():
    model, _ = _dp(unit="patient", clip=0.05)
    model.chunk = 5                            # smaller than one patient
    rows = np.arange(24)
    patients = rows // 6
    tot = model._clipped_sum(rows, patients)   # must not raise
    norm = torch.sqrt(sum((v ** 2).sum() for v in tot.values()))
    assert float(norm) <= 4 * 0.05 * (1 + 1e-4)


def test_padding_rows_contribute_nothing():
    """Chunks are padded to a fixed size with copies of their first row;
    the padding unit must be dropped exactly."""
    model, _ = _dp(unit="patient", clip=1e9)
    params = dict(model.model.named_parameters())

    def fake(rows, p=None):          # gradient of row i is the constant i
        r = torch.as_tensor(rows, dtype=torch.float32)
        return {k: r.view(-1, *([1] * v.dim())).expand(len(rows), *v.shape)
                .clone() for k, v in params.items()}

    model._per_row_grads = fake
    rows = np.arange(1, 11)                  # 10 rows, 5 patients, chunk 64
    total = model._clipped_sum(rows, rows // 2)
    for v in total.values():                 # 1 + ... + 10, not 55 + 54
        assert torch.allclose(v, torch.full_like(v, 55.0))
    model.chunk = 4                          # several chunks, all padded
    total = model._clipped_sum(rows, rows // 2)
    for v in total.values():
        assert torch.allclose(v, torch.full_like(v, 55.0))


def test_true_posteriors_are_distributions():
    model, _ = _dp()
    p = model._noised(np.arange(50))
    pos = 0
    for k in model.sizes:
        s = p["log_true"][:, pos:pos + k].exp().sum(1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
        pos += k


def test_terminal_marginal_is_uniform():
    model, _ = _dp()
    s = model.sch
    T = s.T
    # q(x_T | x_0) = abar_T * x0 + (1 - abar_T)/K  ~ uniform
    assert float(s.abar[T]) < 1e-3


def test_accountant_hits_target_epsilon():
    df = _toy(2000)
    m = DPTabDDPM(hidden=(16,), emb=16, timesteps=20, expected_batch=200,
                  epochs=2, target_epsilon=4.0, seed=0, verbose=False)
    m.fit(df, ["g", "h"])
    assert abs(m.epsilon_spent_ - 4.0) < 0.01
    assert m.epsilon_add_remove_ < m.epsilon_spent_


def test_patient_level_fit_runs_and_accounts():
    df = _toy(600)
    groups = np.arange(600) // 3
    m = DPTabDDPM(hidden=(16,), emb=16, timesteps=20, expected_batch=40,
                  epochs=1, target_epsilon=5.0, unit="patient", seed=0,
                  verbose=False)
    m.fit(df, ["g", "h"], groups=groups)
    assert m.n_units_ == 200
    assert abs(m.epsilon_spent_ - 5.0) < 0.01
    out = m.sample(100)
    assert list(out.columns) == list(df.columns)


def test_nonprivate_model_learns_cluster_structure():
    df = _toy(4000)
    m = TabDDPM(hidden=(128, 128), emb=64, timesteps=200, steps=1500,
                batch_size=512, lr=2e-3, seed=0, verbose=False)
    m.fit(df, ["g", "h"])
    syn = m.sample(3000)
    assert set(syn["g"]) <= {"a", "b"} and set(syn["h"]) <= {"u", "v", "w"}
    share_a = (syn["g"] == "a").mean()
    assert abs(share_a - 0.7) < 0.08
    # conditional structure: x must sit in the right cluster for its category
    right = ((syn["g"] == "a") & (syn["x"] < 0)) | \
            ((syn["g"] == "b") & (syn["x"] > 0))
    assert right.mean() > 0.9
    # and the numeric-numeric relation y ~ 0.5 x must survive
    assert np.corrcoef(syn["x"], syn["y"])[0, 1] > 0.9


def test_seg_log_softmax_normalises():
    x = torch.randn(4, 7)
    y = _seg_log_softmax(x, [3, 4]).exp()
    assert torch.allclose(y[:, :3].sum(1), torch.ones(4), atol=1e-6)
    assert torch.allclose(y[:, 3:].sum(1), torch.ones(4), atol=1e-6)


if __name__ == "__main__":          # dependency-free runner (pytest optional)
    import time
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            try:
                fn()
                print(f"PASS  {name}  ({time.time()-t:.1f}s)", flush=True)
            except Exception:
                fails += 1
                print(f"FAIL  {name}", flush=True)
                traceback.print_exc()
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
