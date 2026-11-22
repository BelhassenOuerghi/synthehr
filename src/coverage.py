"""
coverage.py
-----------
Density and coverage for every arm and seed, computed after the main run.

Why this exists, stated plainly: it was added after the results showed that
private generators score recall close to 1.0 with precision close to 0. That
combination is an artefact of improved precision/recall, not coverage: an
over-dispersed synthetic cloud has huge nearest-neighbour balls, which swallow
the real data. Density and coverage (Naeem et al., 2020) use balls around real
points only and are not fooled. It is a secondary realism metric, reported as
added post hoc, and it changes no pre-registered test.

Reads the saved synthetic datasets (pooled releases are rebuilt exactly as
`jobs.py` builds them) and writes results/coverage.json.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

import jobs as J
import realism_data as RD
import settings
from fidelity import density_coverage
from multiseed import _restore

warnings.filterwarnings("ignore")


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


def main():
    cfg = settings.load()
    out_p = os.path.join(os.environ.get("ENGD_RESULTS_DIR",
                                        os.path.join(settings.HERE, "results")),
                         "coverage.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for task in ("readmission", "los"):
        for seed in cfg["seeds"]:
            c = dict(cfg); c["seed"] = seed
            d = RD.load_with_groups(task, c)
            for arm in J.ALL_ARMS:
                key = f"{task}/{seed}/{arm}"
                if key in res or not os.path.exists(
                        J.result_path(task, seed, arm)):
                    continue
                syn = dataset(c, task, seed, arm, d)
                if syn is None:
                    continue
                res[key] = density_coverage(d["train"], d["test"], syn,
                                            d["discrete"], seed=seed)
                print(key, res[key], flush=True)
                json.dump(res, open(out_p, "w"), indent=1)
    print(f"{len(res)} entries -> {out_p}")


if __name__ == "__main__":
    main()
