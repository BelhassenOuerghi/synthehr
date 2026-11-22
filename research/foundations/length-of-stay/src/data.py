"""
data.py, Shared dataset loading, patient-level splitting, and preprocessing.

Both train.py and evaluate.py import from here so they build byte-identical
train/val/test splits and the same preprocessing pipeline. The split is on
`subject_id` (patient-level) to prevent leakage of a patient's repeat ICU
stays across the train/test boundary.
"""

import os

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extract import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES  # noqa: F401

TARGET = "unitlengthstay_trunc"


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(config):
    """Merge cohort + features into a single modelling frame keyed by stay_id."""
    d = config["derived_dir"]
    cohort = pd.read_parquet(f"{d}/cohort.parquet")
    feats = pd.read_parquet(f"{d}/features.parquet")
    cohort.columns = [c.lower() for c in cohort.columns]
    feats.columns = [c.lower() for c in feats.columns]
    df = cohort.merge(feats, on="stay_id", how="inner", suffixes=("", "_f"))
    # Drop any duplicate feature columns carried by both frames.
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def make_splits(df, config):
    """Return a dict of boolean masks for train/val/test, split on subject_id.

    Patients are shuffled once (seed) and partitioned by the configured
    proportions so no subject appears in more than one split.
    """
    sp = config["split"]
    rng = np.random.RandomState(int(sp["random_state"]))
    subjects = df["subject_id"].unique()
    subjects = subjects.copy()
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(sp["train"] * n))
    n_val = int(round(sp["val"] * n))
    train_s = set(subjects[:n_train])
    val_s = set(subjects[n_train:n_train + n_val])
    test_s = set(subjects[n_train + n_val:])

    masks = {
        "train": df["subject_id"].isin(train_s).values,
        "val": df["subject_id"].isin(val_s).values,
        "test": df["subject_id"].isin(test_s).values,
    }
    return masks


def build_preprocessor(scale_numeric):
    """ColumnTransformer: median-impute (+optional scale) numerics, impute+one-hot categoricals.

    scale_numeric=True is used for the linear base learner (scale-sensitive);
    the random forest uses scale_numeric=False.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def xy(df):
    """Split a modelling frame into (X features, y target)."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(float).values
    return X, y
