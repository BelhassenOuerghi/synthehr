"""
data.py
-------
Load the three real prediction-task feature matrices already built by the
replication repositories, and expose them in a uniform form for synthesis and
for the train-on-synthetic / test-on-real (TSTR) evaluation.

Nothing here reads raw MIMIC-IV; it consumes the *derived, de-identified
feature parquets* the repos produced (still governed by the PhysioNet DUA and
never committed).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _p(cfg, key):
    return os.path.join(cfg["repos_root"], cfg[key])


# ----------------------------------------------------------------------------
# LOS: regression (target: truncated ICU length of stay, days)
# ----------------------------------------------------------------------------
def load_los(cfg):
    feats = pd.read_parquet(_p(cfg, "los_features"))
    cohort = pd.read_parquet(_p(cfg, "los_cohort"))[
        ["stay_id", "unitlengthstay_trunc"]]
    splits = pd.read_parquet(_p(cfg, "los_splits"))
    df = feats.merge(cohort, on="stay_id").merge(splits, on="stay_id")
    df = df.rename(columns={"unitlengthstay_trunc": "los_days"})
    split = df["split"].to_numpy()
    df = df.drop(columns=["stay_id", "split"])
    discrete = ["gender", "admission_type", "unitcode"]
    # train = train+val (mirrors the repo's final-fit convention); test untouched
    tr = df[np.isin(split, ["train", "val"])].reset_index(drop=True)
    te = df[split == "test"].reset_index(drop=True)
    return {
        "name": "los", "task": "regression", "target": "los_days",
        "discrete": discrete, "train": tr, "test": te,
    }


# ----------------------------------------------------------------------------
# Readmission: binary classification (target: 30-day failed-discharge label)
# ----------------------------------------------------------------------------
def load_readmission(cfg):
    """Readmission cohort with a PATIENT-LEVEL (group) train/test split.

    The cohort holds 87,041 ICU stays from only 60,537 patients: 24.5% of
    patients have more than one stay, accounting for 47.5% of rows. A row-level
    split therefore leaks, an earlier version of this loader used a stratified
    row split, which put 6,046 patients on both sides and made 42.7% of the test
    set patients the model had already seen. That inflated the TRTR ceiling
    (+0.011 ROC-AUC) and, more seriously, invalidated the membership-inference
    evaluation, whose "non-members" must be patients the generator never saw.

    We therefore split on `subject_id` with GroupShuffleSplit: train and test
    are patient-disjoint by construction. (The LOS task's saved split was
    already patient-level.) The label remains close to stratified because the
    cohort is large and readmission is not strongly clustered within patients;
    the realised prevalence is reported by the caller.
    """
    df = pd.read_parquet(_p(cfg, "readmission_frame"))
    if "subject_id" not in df.columns:
        raise ValueError("readmission frame lacks subject_id, a patient-level "
                         "split cannot be guaranteed; refusing to fall back to "
                         "a leaking row-level split")
    groups = df["subject_id"].to_numpy()
    df = df.drop(columns=[c for c in ["stay_id", "subject_id", "hadm_id"]
                          if c in df.columns])
    df["label"] = df["label"].astype(int).astype(str)   # discrete target
    discrete = ["gender", "admission_type", "admission_location", "insurance",
                "first_careunit", "label"]
    discrete = [c for c in discrete if c in df.columns]

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["test_size"],
                            random_state=cfg["seed"])
    itr, ite = next(gss.split(df, groups=groups))
    tr, te = df.iloc[itr], df.iloc[ite]
    assert not (set(groups[itr]) & set(groups[ite])), "patient leakage in split"
    return {
        "name": "readmission", "task": "classification", "target": "label",
        "discrete": discrete, "train": tr.reset_index(drop=True),
        "test": te.reset_index(drop=True),
    }


# ----------------------------------------------------------------------------
# Bed demand: daily ICU-admission count time series (temporal)
# ----------------------------------------------------------------------------
def load_beddemand(cfg):
    s = pd.read_csv(_p(cfg, "beddemand_series"))
    y = s["count"].to_numpy(dtype=float)
    test_days = cfg["beddemand"]["test_days"]
    return {
        "name": "beddemand", "task": "forecast",
        "y_train": y[:-test_days], "y_test": y[-test_days:], "full": y,
    }


def clean_numeric(df, discrete):
    """Median-impute numeric NaNs so the GAN sees complete rows (documented)."""
    df = df.copy()
    for c in df.columns:
        if c not in discrete:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    return df
