"""The command line runs end to end on fabricated data."""
from __future__ import annotations

import os
import tempfile

from synthehr.cli import main
from synthehr.demo import make_demo_cohort


def test_demo_quick_runs_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        assert main(["demo", "--quick", "--out", d, "--device", "cpu"]) == 0
        assert os.path.exists(os.path.join(d, "tabddpm", "data-card.md"))


def test_fit_sample_evaluate_gate():
    with tempfile.TemporaryDirectory() as d:
        df = make_demo_cohort(600, seed=3)
        df.iloc[:450].to_csv(os.path.join(d, "train.csv"), index=False)
        df.iloc[450:].to_csv(os.path.join(d, "test.csv"), index=False)
        disc = "sex,insurance,admission_type,readmitted_30d"
        model = os.path.join(d, "m.synthehr")
        assert main(["fit", "--data", os.path.join(d, "train.csv"),
                     "--out", model, "--model", "dp-tabddpm", "--discrete",
                     disc, "--group-column", "patient_id", "--privacy-unit",
                     "patient", "--epochs", "1", "--device", "cpu",
                     "--quiet"]) == 0
        syn = os.path.join(d, "syn.csv")
        assert main(["sample", "--model", model, "--rows", "300",
                     "--out", syn]) == 0
        assert main(["evaluate", "--train", os.path.join(d, "train.csv"),
                     "--test", os.path.join(d, "test.csv"), "--synthetic",
                     syn, "--target", "readmitted_30d", "--discrete", disc,
                     "--drop", "patient_id", "--model", model,
                     "--out", os.path.join(d, "rep")]) == 0
        code = main(["gate", "--report", os.path.join(d, "rep", "report.json"),
                     "--evidence", os.path.join(d, "ev")])
        assert code in (0, 2)                     # a decision either way
        assert os.path.exists(os.path.join(d, "ev", "data-card.md"))
