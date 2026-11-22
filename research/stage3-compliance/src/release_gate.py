"""
release_gate.py
---------------
The **programmatic release gate**: the "compliance-vetted release process"
that defines Stage 3's dataset generation. It evaluates the Stage-2 canonical
release candidate against the technical criteria in config.yaml, and
distinguishes two decisions:

  * **internal promotion**, may the candidate advance to the Stage-3 dataset
    folder (`MIMIC-SyntheticPrivacyBiasEthic`) inside this credentialed
    research environment? Decided entirely by the technical gates.
  * **external release**, may it leave the environment? Requires, in
    addition, the HUMAN gates (PhysioNet-DUA/contractual review; practitioner
    governance review), which code must never auto-close. Until both are
    signed, the decision is HOLD, by design, not by omission.

On internal promotion the gate copies the canonical datasets byte-identically,
writes SHA256SUMS (proving Stage 3 changes *process*, not *distribution*), and
emits `release_manifest.json` (provenance chain) plus
`results/release_decision.json` (the audit record).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["los_synthetic.parquet", "readmission_synthetic.parquet",
         "beddemand_synthetic.csv"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gate(cfg):
    g = cfg["gate"]
    s2m = json.load(open(cfg["stage2_manifest"]))
    r2 = json.load(open(cfg["stage2_metrics"]))
    checks = []

    def check(cid, ok, evidence):
        checks.append({"id": cid, "pass": bool(ok), "evidence": evidence})

    # 1, formal privacy budget, per released tabular dataset
    arm = "fairsample"
    for task in ("los", "readmission"):
        spent = s2m["datasets"][task]["arms"][arm]["epsilon_spent"]
        check(f"formal_privacy_{task}", spent <= g["max_epsilon"] + 1e-6,
              f"epsilon spent {spent:.3f} <= {g['max_epsilon']} at "
              f"delta {g['max_delta']}")

    # 2, empirical privacy: MIA in the chance band, near-duplicates bounded
    lo, hi = g["mia_auc_band"]
    for task, m in r2["mia"].items():
        aucs = [m["attack_auc_dcr"], m["attack_auc_knn_classifier"]]
        check(f"empirical_privacy_{task}",
              all(lo <= a <= hi for a in aucs)
              and m["near_duplicate_share@0.1"] <= g["max_near_duplicate"],
              f"attack AUCs {['%.3f' % a for a in aucs]} in [{lo},{hi}]; "
              f"near-dup {m['near_duplicate_share@0.1']:.3f} <= "
              f"{g['max_near_duplicate']}")

    # 3, fairness evidence present: representation audit + disparity metrics
    rep = r2["readmission"][arm]["representation"].get("insurance")
    fair = r2["readmission"][arm]["fairness"].get("insurance")
    check("fairness_evidence", rep is not None and fair is not None,
          "representation audit + disparity metrics computed for the "
          "released arm (interpreted jointly with utility in the DPIA)")

    # 4, honest utility disclosure, including negative results
    neg_disclosed = (r2["los"][arm]["r2"] is not None)   # negative R2 published
    check("utility_disclosure",
          bool(g["require_negative_results_disclosed"]) and neg_disclosed,
          f"TSTR published incl. negative results (LOS R2 "
          f"{r2['los'][arm]['r2']:.3f}, readmission AUC "
          f"{r2['readmission'][arm]['roc_auc']:.3f})")

    # 5, provenance: upstream manifests exist and candidates are on disk
    prov_ok = all(os.path.exists(cfg[k]) for k in
                  ("stage0_manifest", "stage1_manifest", "stage2_manifest"))
    cand = {f: os.path.join(cfg["stage2_root"], f) for f in FILES}
    prov_ok = prov_ok and all(os.path.exists(p) for p in cand.values())
    check("provenance", prov_ok,
          "stage 0/1/2 manifests + canonical candidate files present")

    technical_pass = all(c["pass"] for c in checks)

    # human gates, never auto-closed
    human = [{"id": h["id"], "name": h["name"], "status": "OPEN"}
             for h in g["human_gates"]]

    decision = {
        "candidate": {f: _sha256(p) for f, p in cand.items()},
        "checks": checks,
        "technical_pass": technical_pass,
        "internal_promotion": "PASS" if technical_pass else "FAIL",
        "human_gates": human,
        "external_release": "HOLD (human gates open)" if technical_pass
                            else "FAIL",
    }

    # promote on technical pass: byte-identical copy + checksums + manifest
    if technical_pass:
        out = cfg["release_root"]
        os.makedirs(out, exist_ok=True)
        sums = []
        for f, src in cand.items():
            dst = os.path.join(out, f)
            shutil.copy2(src, dst)
            digest = _sha256(dst)
            assert digest == decision["candidate"][f], "copy corrupted"
            sums.append(f"{digest}  {f}")
        with open(os.path.join(out, "SHA256SUMS.txt"), "w") as fh:
            fh.write("\n".join(sums) + "\n")
        release_manifest = {
            "stage": 3,
            "process": "Stage-2 canonical arm promoted through the Stage-3 "
                       "release gate; data distribution UNCHANGED "
                       "(checksums prove byte-identity with stage-2 "
                       "canonical files)",
            "canonical_arm": arm,
            "upstream": {k: cfg[k] for k in
                         ("stage0_manifest", "stage1_manifest",
                          "stage2_manifest")},
            "decision": {k: decision[k] for k in
                         ("internal_promotion", "external_release")},
            "evidence_package": "doc/evidence/ in the Stage-3 repository",
        }
        with open(os.path.join(out, "release_manifest.json"), "w") as fh:
            json.dump(release_manifest, fh, indent=2)
        print(f"promoted {len(cand)} datasets -> {out} (byte-identical, "
              "checksummed)", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "release_decision.json"), "w") as fh:
        json.dump(decision, fh, indent=2)
    print(json.dumps({k: decision[k] for k in
                      ("internal_promotion", "external_release")}, indent=2),
          flush=True)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['id']}: "
              f"{c['evidence']}", flush=True)
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    args = ap.parse_args()
    with open(args.config) as f:
        run_gate(yaml.safe_load(f))
