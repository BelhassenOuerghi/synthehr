"""Shared fixtures for the test suite. All data is fabricated."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from synthehr.demo import DISCRETE, make_demo_cohort  # noqa: E402

RESEARCH_COMMON = os.path.join(ROOT, "research", "common")
RESEARCH_STAGE4 = os.path.join(ROOT, "research", "stage4-realism", "src")


def toy(n: int = 400, seed: int = 0):
    """A small fabricated table without the identifier column."""
    df = make_demo_cohort(n, seed=seed)
    groups = df.pop("patient_id").to_numpy()
    return df, list(DISCRETE), groups


def research_path():
    """Make the frozen research modules importable."""
    for p in (RESEARCH_COMMON, RESEARCH_STAGE4):
        if p not in sys.path:
            sys.path.append(p)
