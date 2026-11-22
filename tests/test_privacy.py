"""The private trainer computes what it claims: exact per-example gradients,
bounded per-unit contributions, padding that contributes nothing, and an
accountant that hits its targets."""
from __future__ import annotations

import math

import numpy as np
import torch

from _helpers import toy
from synthehr.generators.tabddpm import DPTabDDPM
from synthehr.privacy.accountant import Run, calibrate_sigma, compose, epsilon


def _model(unit="record", clip=1.0, m=3):
    df, disc, _ = toy(300)
    model = DPTabDDPM(size=(32, 32), emb=16, timesteps=50, clip_norm=clip,
                      noise_multiplicity=m, privacy_unit=unit, chunk=64,
                      expected_batch=64, epochs=1, device="cpu", seed=1)
    model._setup(df, disc)
    return model


def test_per_example_gradients_match_an_explicit_loop():
    model = _model()
    rows = np.arange(8)
    p = model._noised(rows, reps=model.m)
    g = model._per_row_grads(rows, p=p)
    for i in rows:
        sl = slice(i * model.m, (i + 1) * model.m)
        pi = {k: v[sl] for k, v in p.items()}
        model.model.zero_grad()
        model._loss(model.model(pi["xin"], pi["t"]), pi).backward()
        for k, prm in model.model.named_parameters():
            assert torch.allclose(g[k][i], prm.grad, atol=1e-5, rtol=1e-4), k


def test_patient_contributions_are_bounded_by_the_clip():
    model = _model(unit="patient", clip=0.05)
    rows = np.arange(40)
    patients = rows // 4
    for pid in range(10):
        sel = rows[patients == pid]
        tot = model._clipped_sum(sel, patients[sel])
        norm = torch.sqrt(sum((v ** 2).sum() for v in tot.values()))
        assert float(norm) <= 0.05 * (1 + 1e-4)


def test_padding_rows_contribute_nothing():
    model = _model(unit="patient", clip=1e9)
    params = dict(model.model.named_parameters())

    def fake(rows, p=None):            # the gradient of row i is the constant i
        r = torch.as_tensor(rows, dtype=torch.float32)
        return {k: r.view(-1, *([1] * v.dim())).expand(len(rows), *v.shape)
                .clone() for k, v in params.items()}

    model._per_row_grads = fake
    rows = np.arange(1, 11)
    for chunk in (64, 4):
        model.chunk = chunk
        total = model._clipped_sum(rows, rows // 2)
        for v in total.values():
            assert torch.allclose(v, torch.full_like(v, 55.0))


def test_accountant_targets_and_composition():
    q, steps, d = 2048 / 30000, 2930, 1e-5
    s = calibrate_sigma(10.0, q, steps, d)
    assert 9.98 < epsilon(q, s, steps, d) <= 10.0
    four = compose([Run(q, s, steps)] * 4, d)
    assert 10.0 < four < 40.0               # Renyi composition is sub-linear
    s4 = calibrate_sigma(10.0, q, steps, d, runs=4)
    assert 9.98 < compose([Run(q, s4, steps)] * 4, d) <= 10.0
    # the plain Gaussian mechanism has the closed-form RDP alpha/(2 sigma^2)
    assert math.isclose(compose([Run(1.0, 2.0, 1)], d, orders=[3]),
                        3 / 8 + math.log(1 / d) / 2)
