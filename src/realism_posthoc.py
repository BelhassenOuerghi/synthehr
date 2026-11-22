"""
realism_posthoc.py
------------------
Recompute every realism metric for every arm and seed, after the main run,
against the real test split imputed with the TRAINING medians.

Why this exists, stated plainly. Two problems were found in the realism
measurement once the main run's results were in:

1. The frozen harness imputes each split with its own median. On readmission,
   albumin_min is 72% missing; in seeds 43 and 46 the training and test
   medians differ (3.2 against 3.1), and the real-versus-real control then
   separates real training rows from real test rows with detection AUC 0.956
   through that one constant. Realism has to be measured against a test split
   imputed the way a deployed model would see it: with the training medians.
   Utility is still scored on the frozen test split, identically for every
   arm, so that it stays comparable with Stages 0 to 2.
2. Improved precision/recall rewards over-dispersed private samples (recall
   close to 1.0 with precision close to 0). Density and coverage (Naeem et al.,
   2020) are added because they are not fooled.

Neither change touches a pre-registered utility or privacy test. The values
computed inside each job are kept in results/arms/ for audit; this file's
output, results/realism.json, is what the tables and figures report.

Pooled releases are rebuilt exactly as jobs.py builds them.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import jobs as J
import realism_data as RD
import settings
from fidelity import density_coverage, fidelity
from multiseed import _restore

warnings.filterwarnings("ignore")
RES = os.environ.get("ENGD_RESULTS_DIR", os.path.join(settings.HERE, "results"))
OUT = os.path.join(RES, "realism.json")


def dataset(cfg, task, seed, arm, d):
    n, disc = len(d["train"]), d["discrete"]
    if arm == "TRTR":
        return d["train"]
    if arm in J.POOLED:
        comps = J.POOLED[arm]
        if not all(os.path.exists(J.data_path(cfg, task, seed, c))
                   for c in comps):
            return None
        per = [n // len(comps) + (1 if i < n % len(comps) else 0)
               for i in range(len(comps))]
        syn = pd.concat([pd.read_parquet(J.data_path(cfg, task, seed, c))[:k]
                         for c, k in zip(comps, per)], ignore_index=True)
        rng = np.random.default_rng(seed)
        syn = syn.iloc[rng.permutation(len(syn))].reset_index(drop=True)
        return _restore(syn, d["train"], disc)
    p = J.data_path(cfg, task, seed, arm)
    if not os.path.exists(p):
        return None
    return _restore(pd.read_parquet(p), d["train"], disc)


def one_group(args):
    task, seed, done = args
    import torch
    torch.set_num_threads(2)
    cfg = settings.load()
    cfg["seed"] = seed
    d = RD.load_with_groups(task, cfg)
    out = {}
    for arm in J.ALL_ARMS:
        key = f"{task}/{seed}/{arm}"
        if key in done or not os.path.exists(J.result_path(task, seed, arm)):
            continue
        syn = dataset(cfg, task, seed, arm, d)
        if syn is None:
            continue
        r = fidelity(d["train"], d["test_fid"], syn, d["discrete"],
                     d["target"], seed=seed)
        r.update(density_coverage(d["train"], d["test_fid"], syn,
                                  d["discrete"], seed=seed))
        out[key] = r
        print(key, {k: round(v, 3) for k, v in r.items()
                    if isinstance(v, float)}, flush=True)
    return out


def main(workers=4):
    cfg = settings.load()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    groups = [(t, s, set(res)) for t in ("readmission", "los")
              for s in cfg["seeds"]]
    with Pool(workers) as pool:
        for part in pool.imap_unordered(one_group, groups):
            res.update(part)
            json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"{len(res)} entries -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
