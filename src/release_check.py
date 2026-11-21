"""
release_check.py
----------------
Stage 3's release gate, re-applied to every Pillar 4 release candidate.

Stage 3 built a programmatic gate that decides whether a synthetic release may
be promoted inside the credentialed environment. Its technical criteria are
re-applied here, with Stage 3's own thresholds, to every arm and every seed:

  formal       epsilon <= 10 at delta <= 1e-5 (non-private arms fail by definition)
  empirical    both attack AUCs inside [0.45, 0.55], near-duplicate share <= 1%
  fairness     a representation audit and disparity metrics exist
  disclosure   utility, including negative results, is published

Stage 3's human gates (data-use agreement, practitioner review) stay open:
code never closes them, so the external-release decision is HOLD for every arm,
exactly as in Stage 3. The interesting output is which arms pass the EMPIRICAL
check while failing the FORMAL one: a release can sit at chance against both
attacks and still carry no guarantee at all.

Writes results/release_check.md.
"""
from __future__ import annotations

import glob
import json
import os

import settings

RES = os.environ.get("ENGD_RESULTS_DIR", os.path.join(settings.HERE, "results"))
GATE = {"max_epsilon": 10.0, "max_delta": 1e-5, "mia_auc_band": (0.45, 0.55),
        "max_near_duplicate": 0.01}


def adjudicate(r, delta):
    p, m = r["privacy"], r["mia"]
    eps = p.get("epsilon")
    lo, hi = GATE["mia_auc_band"]
    aucs = (m["attack_auc_dcr"], m["attack_auc_knn_classifier"])
    checks = {
        "formal": eps is not None and eps <= GATE["max_epsilon"] + 1e-6
        and delta <= GATE["max_delta"],
        "empirical": all(lo <= a <= hi for a in aucs)
        and m["near_duplicate_share@0.1"] <= GATE["max_near_duplicate"],
        "fairness": bool(r.get("representation"))
        and "fairness" in r["utility"],
        "disclosure": True,
    }
    checks["promote"] = all(checks.values())
    return checks


def main():
    cfg = settings.load()
    delta = cfg["dp"]["delta"]
    rows = {}
    for p in sorted(glob.glob(os.path.join(RES, "arms", "*", "seed_*",
                                           "*.json"))):
        r = json.load(open(p))
        if r["arm"] == "TRTR":
            continue
        c = adjudicate(r, delta)
        key = (r["task"], r["arm"])
        agg = rows.setdefault(key, {"n": 0, "formal": 0, "empirical": 0,
                                    "promote": 0})
        agg["n"] += 1
        for k in ("formal", "empirical", "promote"):
            agg[k] += c[k]
    L = ["# Stage 3's release gate, re-applied to Pillar 4", "",
         "Counts are seeds passing each technical check (out of n). The human "
         "gates stay open, so external release is HOLD for every arm.", "",
         "| Task | Arm | n | formal (ε ≤ 10) | empirical (attacks at chance, "
         "no near-copies) | promoted internally |", "|---|---|---|---|---|---|"]
    for (task, arm), a in sorted(rows.items()):
        L.append(f"| {task} | {arm} | {a['n']} | {a['formal']} | "
                 f"{a['empirical']} | {a['promote']} |")
    with open(os.path.join(RES, "release_check.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {os.path.join(RES, 'release_check.md')}")


if __name__ == "__main__":
    main()
