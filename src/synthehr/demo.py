"""
A fabricated ICU cohort for trying the tool without access to real data.

Every value is simulated: no record corresponds to a real person. The
structure mimics what makes health tables hard to synthesise: repeat stays
of the same patient, skewed laboratory values, a rare category, and an
outcome that depends on several features at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISCRETE = ["sex", "insurance", "admission_type", "readmitted_30d"]
SENSITIVE = ["sex", "insurance"]
TARGET = "readmitted_30d"


def make_demo_cohort(n_stays: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Simulated ICU stays: one row per stay, ``patient_id`` groups repeat
    stays, ``readmitted_30d`` is a binary outcome ("1"/"0")."""
    rng = np.random.default_rng(seed)
    n_pat = int(n_stays / 1.4)
    pid = np.concatenate([np.arange(n_pat),
                          rng.integers(0, n_pat, n_stays - n_pat)])
    rng.shuffle(pid)
    pat_age = rng.normal(64, 16, n_pat).clip(18, 95)
    pat_sex = rng.choice(["F", "M"], n_pat, p=[0.44, 0.56])
    pat_ins = rng.choice(["Medicare", "Private", "Medicaid", "Other", "None"],
                         n_pat, p=[0.52, 0.27, 0.14, 0.06, 0.01])
    age = pat_age[pid] + rng.uniform(0, 2, n_stays)
    elderly = (age > 75).astype(float)
    adm = rng.choice(["emergency", "urgent", "elective"], n_stays,
                     p=[0.68, 0.18, 0.14])
    hr = rng.normal(88, 14, n_stays) + 6 * (adm == "emergency")
    rr = rng.normal(19, 4, n_stays)
    sbp = rng.normal(118, 18, n_stays) - 4 * elderly
    creat = np.exp(rng.normal(0.1, 0.55, n_stays)) + 0.25 * elderly
    lactate = np.exp(rng.normal(0.4, 0.5, n_stays))
    wbc = np.exp(rng.normal(2.3, 0.35, n_stays))
    prior = rng.poisson(0.6 + 0.8 * elderly)
    los = np.exp(rng.normal(0.8, 0.7, n_stays) + 0.25 * np.log1p(lactate)
                 + 0.15 * (adm == "emergency"))
    logit = (-2.4 + 0.025 * (age - 64) + 0.55 * np.log(creat)
             + 0.35 * prior + 0.4 * (pat_ins[pid] == "Medicaid")
             + 0.25 * np.log1p(lactate) - 0.3 * (adm == "elective")
             + 0.08 * los)
    readm = (rng.random(n_stays) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "patient_id": pid, "age": age.round(), "sex": pat_sex[pid],
        "insurance": pat_ins[pid], "admission_type": adm,
        "heart_rate": hr.round(1), "resp_rate": rr.round(1),
        "systolic_bp": sbp.round(1), "creatinine": creat.round(2),
        "lactate": lactate.round(2), "wbc": wbc.round(1),
        "prior_admissions": prior, "icu_los_days": los.round(2),
        "readmitted_30d": readm.astype(str)})
