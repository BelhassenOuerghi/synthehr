"""Measurement behaves as documented on cases with a known answer."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from _helpers import toy
from synthehr import Report, evaluate
from synthehr.data import impute_with_train_medians, patient_disjoint_split
from synthehr.demo import SENSITIVE, TARGET, make_demo_cohort
from synthehr.evaluation.realism import detection_auc
from synthehr.fairness import fair_resample


def _split(n=1500):
    df = make_demo_cohort(n, seed=0)
    train, test = patient_disjoint_split(df, df["patient_id"], seed=0)
    return train.drop(columns="patient_id"), test.drop(columns="patient_id")


def test_real_versus_real_is_hard_and_noise_is_easy():
    train, test = _split()
    disc = ["sex", "insurance", "admission_type", TARGET]
    assert detection_auc(test, train, disc, seed=0) < 0.6
    noise = train.copy()
    for c in ("heart_rate", "creatinine", "lactate"):
        noise[c] = np.random.default_rng(0).permutation(noise[c].to_numpy()) * 3
    assert detection_auc(test, noise, disc, seed=0) > 0.9


def test_evaluate_report_roundtrip_and_ceiling():
    train, test = _split()
    disc = ["sex", "insurance", "admission_type", TARGET]
    rep = evaluate(train, test, train.copy(), target=TARGET,
                   discrete_columns=disc, sensitive=SENSITIVE,
                   n_candidates=300, min_group_size=30)
    # releasing the real data itself (same rows, same order, so the downstream
    # model is identical): full utility, and the attack catches it
    assert abs(rep.utility_ratio() - 1.0) < 1e-9
    assert rep.privacy["attack_auc_dcr"] > 0.95
    assert rep.privacy["epsilon"] is None
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.json")
        rep.to_json(p)
        assert Report.from_json(p).to_dict() == rep.to_dict()
    assert "Utility" in rep.to_markdown()


def test_imputation_uses_training_medians():
    df, disc, _ = toy(200)
    a, b = df.copy(), df.copy()
    b.loc[:50, "lactate"] = np.nan
    _, b2 = impute_with_train_medians(a, b, discrete=disc)
    assert (b2.loc[:50, "lactate"] == a["lactate"].median()).all()


def test_patient_disjoint_split():
    df = make_demo_cohort(800, seed=1)
    tr, te = patient_disjoint_split(df, df["patient_id"], seed=0)
    assert not set(tr["patient_id"]) & set(te["patient_id"])


def test_fair_resample_hits_targets():
    df = make_demo_cohort(6000, seed=2)
    out, diag = fair_resample(df, 2000, ["sex"], 0.02)
    assert abs(diag["achieved"]["sex"]["F"] - 0.5) < 2e-3
    assert diag["pool_sufficient"]
