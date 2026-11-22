"""
test_gate.py
------------
Negative controls for the release gate. A compliance gate that cannot FAIL is
theatre; these tests prove this one can, and that a failing candidate is never
promoted to the release folder.

  1. over-budget ε  -> formal-privacy check FAILs, nothing promoted;
  2. attack AUC outside the chance band -> empirical-privacy check FAILs;
  3. the real candidate PASSes and is promoted (positive control).

Run via `make test`.
"""
from __future__ import annotations

import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_gate as G

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = "/tmp/engd_gate_negcontrol"


def main():
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    # 1, over-budget epsilon must fail and must not promote
    bad = copy.deepcopy(cfg)
    bad["gate"]["max_epsilon"] = 1.0
    bad["release_root"] = TMP + "_eps"
    d = G.run_gate(bad)
    assert d["internal_promotion"] == "FAIL", "over-budget candidate passed!"
    assert d["external_release"] == "FAIL"
    assert not os.path.exists(bad["release_root"]), "promoted despite FAIL!"
    print("neg-control 1 ok: over-budget eps FAILs, no promotion")

    # 2, attacks outside the band must fail
    bad2 = copy.deepcopy(cfg)
    bad2["gate"]["mia_auc_band"] = [0.499, 0.501]
    bad2["release_root"] = TMP + "_mia"
    d2 = G.run_gate(bad2)
    assert d2["internal_promotion"] == "FAIL", "leaky candidate passed!"
    assert not os.path.exists(bad2["release_root"])
    print("neg-control 2 ok: attack outside chance band FAILs, no promotion")

    # 3, positive control: the real candidate passes, external stays HELD
    d3 = G.run_gate(cfg)
    assert d3["internal_promotion"] == "PASS", "real candidate should pass"
    assert d3["external_release"].startswith("HOLD"), \
        "external release must never auto-open"
    assert all(h["status"] == "OPEN" for h in d3["human_gates"])
    print("positive control ok: real candidate PASSes internally; "
          "external release HELD on human gates")

    print("GATE TESTS PASSED, the gate is falsifiable, and human gates "
          "cannot be closed by code.")


if __name__ == "__main__":
    main()
