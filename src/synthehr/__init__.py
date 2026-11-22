"""
SynthEHR: synthetic tabular health data with measured realism, formal privacy
and auditable release evidence.

Generate::

    from synthehr import TabDDPM, DPTabDDPM

    gen = TabDDPM().fit(train, discrete_columns=["sex", "insurance", "label"])
    synthetic = gen.sample(10_000)

    private = DPTabDDPM(epsilon=10, privacy_unit="patient").fit(
        train, discrete_columns=[...], groups=train["patient_id"])

Measure::

    from synthehr import evaluate
    report = evaluate(real_train, real_test, synthetic, target="label",
                      discrete_columns=[...], sensitive=["sex"])
    print(report.to_markdown())

Decide::

    from synthehr import ReleaseGate, build_evidence_pack
    decision = ReleaseGate().check(report)
    build_evidence_pack(report, decision, "evidence/")
"""
from __future__ import annotations

from synthehr._version import __version__
from synthehr.compliance import ReleaseGate, build_evidence_pack
from synthehr.ensemble import pool
from synthehr.evaluation import Report, evaluate
from synthehr.generators import (CTGAN, DPCTGAN, DPTabDDPM, PoissonAR,
                                 TabDDPM, load)

__author__ = "Belhassen Ouerghi"

__all__ = ["CTGAN", "DPCTGAN", "TabDDPM", "DPTabDDPM", "PoissonAR", "load",
           "pool", "evaluate", "Report", "ReleaseGate", "build_evidence_pack",
           "__version__"]
