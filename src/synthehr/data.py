"""
Data preparation that the research showed matters.

* :func:`patient_disjoint_split`: a train/test split in which no patient
  appears on both sides. With repeated stays, a row-level split leaks: in the
  research cohort 43% of test rows belonged to a patient the model had seen,
  which invalidated the membership-inference control group.
* :func:`impute_with_train_medians`: fill missing numerics in EVERY frame
  with the TRAINING medians. Imputing each split with its own median makes
  real training and test rows distinguishable through the imputed constant
  (the research found a real-versus-real detection AUC of 0.956 from one 72%
  missing column).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def patient_disjoint_split(df: pd.DataFrame, groups: Optional[Iterable] = None,
                           test_size: float = 0.2, seed: int = 0):
    """Return ``(train, test)``; patient-disjoint when ``groups`` is given."""
    if groups is None:
        tr, te = train_test_split(df, test_size=test_size, random_state=seed)
        return tr.reset_index(drop=True), te.reset_index(drop=True)
    g = np.asarray(list(groups))
    itr, ite = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=seed).split(df, groups=g))
    assert not set(g[itr]) & set(g[ite])
    return (df.iloc[itr].reset_index(drop=True),
            df.iloc[ite].reset_index(drop=True))


def impute_with_train_medians(train: pd.DataFrame, *others: pd.DataFrame,
                              discrete: Sequence[str] = ()):
    """Median-impute numeric columns of ``train`` and of every other frame
    with the medians of ``train``; categorical gaps become the string
    ``"missing"``. Returns the frames in the order given."""
    med = {c: pd.to_numeric(train[c], errors="coerce").median()
           for c in train.columns if c not in discrete}
    out = []
    for df in (train,) + others:
        df = df.copy()
        for c in df.columns:
            if c in discrete:
                df[c] = df[c].astype(object).where(df[c].notna(), "missing")
            elif c in med:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med[c])
        out.append(df)
    return out[0] if not others else tuple(out)
