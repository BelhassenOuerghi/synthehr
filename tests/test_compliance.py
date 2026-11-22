"""The release gate can refuse, never closes the human gates itself, and the
evidence pack renders every value from the measurements."""
from __future__ import annotations

import os
import tempfile

from synthehr import ReleaseGate, Report, build_evidence_pack


def _report(epsilon=9.9, auc=0.5, util=0.70, degenerate=False):
    return Report(
        target="y", task="classification", n_train=1000, n_test=200,
        n_synthetic=1000, utility={"roc_auc": util, "pr_auc": 0.3,
                                   "accuracy": 0.7},
        utility_ceiling={"roc_auc": 0.78}, realism={
            "detection_auc": 0.7, "coverage": 0.9, "precision": 0.8,
            "marginal_w1": 0.1, "marginal_tvd": 0.05},
        realism_control={"detection_auc": 0.52, "coverage": 0.96},
        privacy={"attack_auc_dcr": auc, "attack_auc_knn_classifier": auc,
                 "tpr@fpr=0.01": 0.01, "near_duplicate_share@0.1": 0.0,
                 "epsilon": epsilon, "delta": 1e-5 if epsilon else None},
        fairness={"sex": {"auc_gap": 0.05, "dp_diff": 0.1, "eo_diff": 0.1,
                          "degenerate": degenerate}})


def test_private_release_is_promoted_but_held():
    d = ReleaseGate().check(_report())
    assert d.internal == "PROMOTE" and d.external == "HOLD"
    assert len(d.open_human_gates) == 2


def test_non_private_release_is_refused_even_with_attacks_at_chance():
    d = ReleaseGate().check(_report(epsilon=None))
    assert d.internal == "REFUSE"
    assert not next(c for c in d.checks if c.id == "formal_privacy").passed
    assert next(c for c in d.checks if c.id == "empirical_privacy").passed


def test_gate_refuses_leaks_degenerate_models_and_useless_releases():
    assert ReleaseGate().check(_report(auc=0.7)).internal == "REFUSE"
    assert ReleaseGate().check(_report(degenerate=True)).internal == "REFUSE"
    useless = _report(util=0.52)
    assert ReleaseGate().check(useless).internal == "PROMOTE"
    assert ReleaseGate(min_utility_ratio=0.5).check(useless).internal == \
        "REFUSE"


def test_external_release_needs_both_human_signatures():
    gate = ReleaseGate()
    assert gate.check(_report(), ["data_use"]).external == "HOLD"
    assert gate.check(_report(), ["data_use", "governance"]).external == \
        "RELEASE"


def test_evidence_pack_injects_every_value():
    rep = _report()
    with tempfile.TemporaryDirectory() as d:
        values = build_evidence_pack(rep, ReleaseGate().check(rep), d,
                                     name="test", date="2026-01-01")
        card = open(os.path.join(d, "data-card.md")).read()
        assert "$" not in card                     # nothing left unfilled
        assert "9.90" in card and values["util"] == "0.700"
        assert {"data-card.md", "release-decision.md", "evidence.json"} <= \
            set(os.listdir(d))
