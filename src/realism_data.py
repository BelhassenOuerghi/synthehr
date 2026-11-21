"""
realism_data.py
---------------
The frozen Stage 0-2 loaders (`data.py`, vendored unchanged) drop the patient
identifier before returning the training frame. Pillar 4 needs it for one
reason only: patient-level differential privacy, where all of a patient's ICU
stays form a single privacy unit.

`load_with_groups` therefore reproduces each frozen loader with the patient id
kept aside, and **asserts that the training and test frames are identical to the
frozen loader's**, so that every Pillar 4 number is computed on exactly the
rows that Stages 0-2 used. It also returns the 30,000-row fitting subsample
that the frozen harness's `_fit_df` draws, with the matching patient ids.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import data as D


def _p(cfg, key):
    return os.path.join(cfg["repos_root"], cfg[key])


def _los_groups(cfg):
    feats = pd.read_parquet(_p(cfg, "los_features"))
    cohort = pd.read_parquet(_p(cfg, "los_cohort"))[
        ["stay_id", "subject_id", "unitlengthstay_trunc"]]
    splits = pd.read_parquet(_p(cfg, "los_splits"))
    df = feats.merge(cohort, on="stay_id").merge(splits, on="stay_id")
    split = df["split"].to_numpy()
    sid = df["subject_id"].to_numpy()
    tr = np.isin(split, ["train", "val"])
    te = split == "test"
    return sid[tr], sid[te]


def _readm_groups(cfg):
    df = pd.read_parquet(_p(cfg, "readmission_frame"))
    groups = df["subject_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["test_size"],
                            random_state=cfg["seed"])
    itr, ite = next(gss.split(df, groups=groups))
    return groups[itr], groups[ite]


def load_with_groups(task, cfg):
    """Frozen-loader output plus `train_groups`, `test_groups`, and the
    fitting subsample (`fit`, `fit_groups`) used by every generator."""
    loader = {"los": D.load_los, "readmission": D.load_readmission}[task]
    ds = loader(cfg)
    g_tr, g_te = (_los_groups if task == "los" else _readm_groups)(cfg)
    assert len(g_tr) == len(ds["train"]) and len(g_te) == len(ds["test"])
    assert not (set(g_tr) & set(g_te)), "train and test share a patient"
    train = D.clean_numeric(ds["train"], ds["discrete"])
    test = D.clean_numeric(ds["test"], ds["discrete"])

    # the frozen harness fits generators on `train.sample(n, random_state=seed)`
    n = cfg["ctgan"]["max_train_rows"]
    if len(train) > n:
        pos = train.sample(n, random_state=cfg["seed"]).index.to_numpy()
    else:
        pos = np.arange(len(train))
    fit = train.loc[pos].reset_index(drop=True)
    out = dict(ds)
    out.update(train=train, test=test, train_groups=g_tr, test_groups=g_te,
               fit=fit, fit_groups=g_tr[pos])
    return out


def check_against_frozen(task, cfg):
    """Re-derive the frozen loader's frames and confirm they match exactly."""
    loader = {"los": D.load_los, "readmission": D.load_readmission}[task]
    ref = loader(cfg)
    mine = load_with_groups(task, cfg)
    for part in ("train", "test"):
        a = D.clean_numeric(ref[part], ref["discrete"])
        pd.testing.assert_frame_equal(a, mine[part])
    return True
