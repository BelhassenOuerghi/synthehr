"""
The library is the research code: for a fixed seed, every generator, the
accountant, the attacks and the realism metrics reproduce the frozen research
implementations bit for bit. This is what lets the published numbers in
research/ vouch for the tool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _helpers import research_path, toy

research_path()

TINY_CTGAN = dict(embedding_dim=16, generator_dim=(16,), discriminator_dim=(16,),
                  batch_size=100, pac=10, vgm_components=3, seed=3,
                  verbose=False)


def test_ctgan_identical_to_research():
    import ctgan as research
    from synthehr import CTGAN
    df, disc, _ = toy()
    a = research.CTGAN(epochs=2, **TINY_CTGAN).fit(df, disc).sample(250)
    b = CTGAN(epochs=2, **TINY_CTGAN).fit(df, disc).sample(250)
    pd.testing.assert_frame_equal(a, b)


def test_dpctgan_identical_to_stage1():
    import dp_ctgan as research
    from synthehr import DPCTGAN
    df, disc, _ = toy()
    # sample straight after each fit: CTGAN sampling draws from torch's
    # global generator, which the other model's fit would advance
    ra = research.DPCTGAN(target_epsilon=8.0, target_delta=1e-5, epochs=2,
                          **TINY_CTGAN).fit(df, disc)
    a = ra.sample(250)
    lb = DPCTGAN(epsilon=8.0, delta=1e-5, epochs=2, **TINY_CTGAN).fit(df, disc)
    b = lb.sample(250)
    pd.testing.assert_frame_equal(a, b)
    assert ra.sigma_ == lb.sigma_
    assert abs(ra.epsilon_spent_ - lb.epsilon_) < 1e-12


def test_tabddpm_identical_to_research():
    import tabddpm as research
    from synthehr import TabDDPM
    df, disc, _ = toy()
    kw = dict(emb=16, timesteps=20, steps=40, batch_size=64, device="cpu",
              seed=5, verbose=False)
    a = research.TabDDPM(hidden=(32, 32), **kw).fit(df, disc).sample(200)
    b = TabDDPM(size=(32, 32), **kw).fit(df, disc).sample(200)
    pd.testing.assert_frame_equal(a, b)


def test_dptabddpm_record_identical_to_research():
    _dptabddpm_identical("record")


def test_dptabddpm_patient_identical_to_research():
    _dptabddpm_identical("patient")


def _dptabddpm_identical(unit):
    import tabddpm as research
    from synthehr import DPTabDDPM
    df, disc, groups = toy()
    kw = dict(emb=16, timesteps=20, clip_norm=1.0, epochs=1,
              expected_batch=64, noise_multiplicity=2, chunk=64, lr=2e-3,
              device="cpu", seed=5, verbose=False)
    a = research.DPTabDDPM(hidden=(32, 32), target_epsilon=6.0,
                           target_delta=1e-5,
                           unit="stay" if unit == "record" else "patient",
                           **kw).fit(df, disc, groups=groups)
    b = DPTabDDPM(size=(32, 32), epsilon=6.0, delta=1e-5, privacy_unit=unit,
                  **kw).fit(df, disc, groups=groups)
    pd.testing.assert_frame_equal(a.sample(200), b.sample(200))
    assert a.sigma_ == b.sigma_ and a.steps_ == b.steps_
    assert abs(a.epsilon_spent_ - b.epsilon_) < 1e-12


def test_poisson_ar_identical_to_research():
    import beddemand_gen as research
    import dp_beddemand as research_dp
    from synthehr import PoissonAR
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0 + np.sin(np.arange(400) / 7.0), size=400).astype(float)
    m = PoissonAR(window=2).fit(y)
    assert np.array_equal(research.fit_poisson_ar(y, 2), m.beta_)
    assert np.array_equal(
        research.sample_series(m.beta_, y[:2], 50, 2,
                               np.random.default_rng(1)),
        m.sample(50, start=y[:2], rng=np.random.default_rng(1)))
    assert np.array_equal(research.forecast(m.beta_, y, 14, 2),
                          m.forecast(y, 14))
    beta, eps, _ = research_dp.dp_fit_poisson_ar(y, 2, target_epsilon=5.0,
                                                 seed=4, verbose=False)
    dp = PoissonAR(window=2, epsilon=5.0, seed=4).fit(y)
    assert np.array_equal(beta, dp.beta_) and abs(eps - dp.epsilon_) < 1e-12


def test_accountant_identical_to_research():
    import privacy_accountant as research
    from synthehr.privacy import accountant as lib
    for q, s, t in ((0.01, 1.1, 5000), (0.068, 2.247, 2930), (1.0, 32.0, 1)):
        assert research.compose_epsilon(q, s, t, 1e-5) == \
            lib.epsilon(q, s, t, 1e-5)
    assert research.calibrate_sigma(10.0, 0.0341, 2930, 1e-5) == \
        lib.calibrate_sigma(10.0, 0.0341, 2930, 1e-5)


def test_attacks_and_realism_identical_to_research():
    import fidelity as research_fid
    from membership_inference import run_mia
    from synthehr.evaluation.realism import realism
    from synthehr.privacy import membership_inference
    df, disc, _ = toy(600)
    train, test = df.iloc[:400].reset_index(drop=True), \
        df.iloc[400:].reset_index(drop=True)
    syn = toy(500, seed=9)[0]
    mia_disc = [c for c in disc if c != "readmitted_30d"]
    lib = membership_inference(train, test, syn, mia_disc, n_candidates=150,
                               seed=2)
    ref = run_mia(train, test, syn, mia_disc, n_candidates=150, seed=2)
    ref.update(research_fid.mia_tail(train, test, syn, mia_disc,
                                     n_candidates=150, seed=2))
    for k, v in ref.items():
        if k == "dcr_auc":
            assert v == lib["attack_auc_dcr"]
        else:
            assert v == lib[k], k
    a = research_fid.fidelity(train, test, syn, disc, "readmitted_30d", seed=1)
    b = realism(train, test, syn, disc, "readmitted_30d", seed=1)
    for k, v in a.items():
        assert v == b[k], k
