"""Generator behaviour: the fit/sample/save contract, privacy accounting,
and that a saved model carries no training rows."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import torch

from _helpers import toy
from synthehr import CTGAN, DPCTGAN, DPTabDDPM, TabDDPM, load, pool
from synthehr.demo import SENSITIVE

SMALL_DDPM = dict(emb=16, timesteps=20, device="cpu", seed=1)


def _check_sample(syn, df, disc):
    assert list(syn.columns) == list(df.columns)
    for c in disc:
        assert set(syn[c].astype(str)) <= set(df[c].astype(str)), c
    assert not syn.isna().any().any()


def test_tabddpm_contract_and_roundtrip():
    df, disc, _ = toy()
    m = TabDDPM(size=(32, 32), steps=60, batch_size=64, **SMALL_DDPM)
    m.fit(df, disc)
    syn = m.sample(150)
    _check_sample(syn, df, disc)
    with tempfile.TemporaryDirectory() as d:
        path = m.save(os.path.join(d, "m.synthehr"))
        pd.testing.assert_frame_equal(load(path).sample(150), syn)


def test_trained_model_does_not_carry_training_rows():
    df, disc, groups = toy()
    for m in (TabDDPM(size=(16,), steps=5, batch_size=32, **SMALL_DDPM),
              DPTabDDPM(size=(16,), epochs=1, expected_batch=32, chunk=32,
                        **SMALL_DDPM)):
        m.fit(df, disc, groups=groups)
        assert m.x_num is None and m.x_cat is None and m.model is None
    c = CTGAN(embedding_dim=8, generator_dim=(8,), discriminator_dim=(8,),
              batch_size=50, epochs=1, vgm_components=2).fit(df, disc)
    assert c.D is None and c.cond.cat_rows is None


def test_dp_tabddpm_accounting_and_patient_unit():
    df, disc, groups = toy(600)
    m = DPTabDDPM(size=(16,), epsilon=4.0, epochs=2, expected_batch=40,
                  chunk=64, privacy_unit="patient", **SMALL_DDPM)
    m.fit(df, disc, groups=groups)
    assert abs(m.epsilon_ - 4.0) < 0.01
    assert m.epsilon_add_remove_ < m.epsilon_
    assert m.n_units_ == len(set(groups))
    try:
        DPTabDDPM(privacy_unit="patient", **SMALL_DDPM).fit(df, disc)
        raise AssertionError("patient-level fit without groups must fail")
    except ValueError:
        pass


def test_dpctgan_fair_sampling_and_single_release():
    df, disc, _ = toy(500)
    m = DPCTGAN(epsilon=8.0, epochs=1, embedding_dim=16, generator_dim=(16,),
                discriminator_dim=(16,), batch_size=100, vgm_components=3,
                fair_train_columns=["sex"], seed=2).fit(df, disc)
    # one count release only: the accountant charges exactly one
    assert m.privacy_.sigma_hist == m.hist_noise_multiplier
    assert abs(m.epsilon_ - 8.0) < 0.01
    fair = m.sample_balanced(2000, SENSITIVE)
    _check_sample(fair, df, disc)
    share = fair["sex"].value_counts(normalize=True)
    assert abs(share.max() - 0.5) < 0.1


def test_pool_composes_budgets():
    df, disc, _ = toy()
    gens = [DPTabDDPM(size=(16,), epsilon=5.0, epochs=1, expected_batch=40,
                      chunk=64, emb=16, timesteps=10, device="cpu",
                      seed=s).fit(df, disc) for s in (1, 2)]
    out, eps = pool(gens, 101)
    assert len(out) == 101
    assert 5.0 < eps < 10.0                 # sub-linear Renyi composition


def test_missing_values_are_rejected_with_a_hint():
    df, disc, _ = toy()
    df.loc[0, "lactate"] = np.nan
    try:
        TabDDPM(steps=1, **SMALL_DDPM).fit(df, disc)
        raise AssertionError("expected a ValueError")
    except ValueError as e:
        assert "impute" in str(e)
