"""
pilot.py
--------
The only hyperparameter search in Pillar 4, run once, before the main
experiment, and kept in the repository so the choice is auditable.

Data: seed 42's TRAINING rows only. A patient-disjoint 15% of them is held out
as a validation set; generators are fitted on a 30,000-row subsample of the
rest (the frozen harness's fitting size), and scored on the validation set with
the frozen downstream models. No test set of any seed is touched.

Grid (deliberately small):
  non-private TabDDPM: 5k / 10k / 20k optimiser steps at batch 4096;
  DP-TabDDPM at epsilon = 10: three (expected batch, epochs) settings.

Caveat, stated in the paper: choosing DP hyperparameters on the private data is
itself a (small) leak that the epsilon does not cover (Papernot & Steinke,
2022). It is standard practice and it is disclosed rather than hidden.
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import realism_data as RD
import settings
from fidelity import fidelity
from multiseed import SCORERS, _cols, _restore
from tabddpm import DPTabDDPM, TabDDPM

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRID = [
    ("np_5k", dict(steps=5000)),
    ("np_10k", dict(steps=10000)),
    ("np_20k", dict(steps=20000)),
    ("dp_b1024_e100", dict(dp=True, expected_batch=1024, epochs=100)),
    ("dp_b4096_e100", dict(dp=True, expected_batch=4096, epochs=100)),
    ("dp_b2048_e200", dict(dp=True, expected_batch=2048, epochs=200)),
]


def main(cfg_path, device):
    cfg = settings.load(cfg_path)
    cfg["seed"] = 42
    out_path = os.path.join(HERE, "results", "pilot.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for task in ("los", "readmission"):
        d = RD.load_with_groups(task, cfg)
        tgt, disc = d["target"], d["discrete"]
        sens = cfg["fairness"]["sensitive"][task]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        itr, iva = next(gss.split(d["train"], groups=d["train_groups"]))
        tr, va = d["train"].iloc[itr], d["train"].iloc[iva]
        tr = tr.reset_index(drop=True); va = va.reset_index(drop=True)
        fit = tr.sample(min(30000, len(tr)), random_state=42).reset_index(
            drop=True)
        dc, num = _cols(tr, tgt, disc)
        res.setdefault(task, {})
        if "TRTR_val" not in res[task]:
            res[task]["TRTR_val"] = SCORERS[task](tr, va, tgt, dc, num, 42,
                                                  sens)
        for name, spec in GRID:
            if name in res[task]:
                continue
            spec = dict(spec)
            t = time.time()
            if spec.pop("dp", False):
                m = DPTabDDPM(target_epsilon=10.0, target_delta=1e-5,
                              lr=2e-3, noise_multiplicity=4, device=device,
                              seed=42, verbose=False, **spec)
            else:
                m = TabDDPM(batch_size=4096, lr=1e-3, device=device, seed=42,
                            verbose=False, **spec)
            m.fit(fit, disc)
            syn = _restore(m.sample(len(tr)), tr, disc)
            r = {"utility": SCORERS[task](syn, va, tgt, dc, num, 42, sens),
                 "fidelity": fidelity(tr, va, syn, disc, tgt, seed=0),
                 "seconds": time.time() - t}
            if hasattr(m, "epsilon_spent_") and m.epsilon_spent_:
                r["epsilon"] = m.epsilon_spent_
                r["steps"] = m.steps_
            res[task][name] = r
            key = "r2" if task == "los" else "roc_auc"
            print(f"{task} {name}: {key}={r['utility'][key]:.3f} "
                  f"det={r['fidelity']['detection_auc']:.3f} "
                  f"prec={r['fidelity']['precision']:.3f} "
                  f"rec={r['fidelity']['recall']:.3f} "
                  f"({r['seconds']:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"), indent=2)
    print("pilot complete", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    main(a.config, a.device)
