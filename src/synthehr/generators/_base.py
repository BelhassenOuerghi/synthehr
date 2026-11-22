"""Shared behaviour of every tabular generator: column typing, device
selection, persistence."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

import pandas as pd
import torch

FORMAT_VERSION = 1


def resolve_device(device: str = "auto") -> torch.device:
    """`"auto"` picks CUDA, then Apple MPS, then CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_discrete_columns(df: pd.DataFrame) -> list[str]:
    """Columns treated as categorical when none are given: non-numeric,
    boolean, or numeric with at most two distinct values (binary flags and
    labels). A heuristic: pass ``discrete_columns`` explicitly when you
    know better (coded categories stored as integers, for example)."""
    out = []
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out.append(c)
        elif s.nunique(dropna=True) <= 2:
            out.append(c)
    return out


class TabularGenerator(ABC):
    """Every generator follows the same contract:

    * ``fit(df, discrete_columns=None, groups=None)`` learns from a complete
      (no missing values) DataFrame and returns ``self``;
    * ``sample(n)`` returns a DataFrame with the training columns, in order;
    * ``save(path)`` / :func:`synthehr.load` persist the fitted generator.

    Private generators additionally expose ``epsilon_`` (and the parameters
    the accountant needs in ``privacy_``) after fitting.
    """

    discrete_columns_: list[str]

    @abstractmethod
    def fit(self, df: pd.DataFrame,
            discrete_columns: Optional[Sequence[str]] = None,
            groups: Optional[Iterable] = None) -> "TabularGenerator":
        ...

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        ...

    @staticmethod
    def _check_input(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fit() expects a pandas DataFrame")
        if df.isna().any().any():
            cols = df.columns[df.isna().any()].tolist()
            raise ValueError(
                f"missing values in {cols}; impute first (for example with "
                "synthehr.data.impute_with_train_medians)")

    def save(self, path: str) -> str:
        """Persist the fitted generator (weights, encoders, settings).

        The file is a pickle: load only files you created or trust."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"format": FORMAT_VERSION, "class": type(self).__name__,
                    "model": self}, path)
        return path


def load(path: str) -> TabularGenerator:
    """Load a generator saved with ``.save()``. Pickle-based: only load
    files you created or trust."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a synthehr model file")
    return blob["model"]
