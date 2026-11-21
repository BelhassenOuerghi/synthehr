"""
diagnose_dp.py
--------------
Where does DP-TabDDPM lose its utility? The same private trainer is run three
ways on the pilot's validation split (seed 42 training rows only):

  sampled     Poisson-sampled DP-SGD machinery with clipping and noise
              switched off: the trainer itself, nothing private;
  clipped     per-unit clipping at C = 1, no noise: what clipping alone costs;
  private     clipping and noise at epsilon = 10: the real recipe.

If `sampled` matches non-private TabDDPM, the trainer is sound; the step from
`sampled` to `clipped` is the cost of clipping; the step from `clipped` to
`private` is the cost of the noise. Writes results/dp_diagnostic.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

from sklearn.model_selection import GroupShuffleSplit

import realism_data as RD
import settings
from fidelity import fidelity
from multiseed import SCORERS, _cols, _restore
from tabddpm import DPTabDDPM

warnings.filterwarnings("ignore")

ARMS = {
    "sampled": dict(clip_norm=1e6, noise_multiplier=1e-6),
    "clipped": dict(clip_norm=1.0, noise_multiplier=1e-6),
    "private": dict(clip_norm=1.0),
}


def main(task, device, extra):
    cfg = settings.load()
    cfg["seed"] = 42
    out_p = os.path.join(settings.HERE, "results", "dp_diagnostic.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    d = RD.load_with_groups(task, cfg)
    tgt, disc = d["target"], d["discrete"]
    sens = cfg["fairness"]["sensitive"][task]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    itr, iva = next(gss.split(d["train"], groups=d["train_groups"]))
    tr = d["train"].iloc[itr].reset_index(drop=True)
    va = d["train"].iloc[iva].reset_index(drop=True)
    fit = tr.sample(min(30000, len(tr)), random_state=42).reset_index(drop=True)
    dc, num = _cols(tr, tgt, disc)
    for name, spec in ARMS.items():
        key = f"{task}:{name}:{json.dumps(extra, sort_keys=True)}"
        if key in res:
            continue
        t = time.time()
        m = DPTabDDPM(target_epsilon=10.0, target_delta=1e-5, lr=2e-3,
                      noise_multiplicity=4, device=device, seed=42,
                      verbose=False, **{**dict(expected_batch=1024,
                                               epochs=100), **extra, **spec})
        m.fit(fit, disc)
        syn = _restore(m.sample(len(tr)), tr, disc)
        u = SCORERS[task](syn, va, tgt, dc, num, 42, sens)
        f = fidelity(tr, va, syn, disc, tgt, seed=0)
        res[key] = {"utility": u, "fidelity": f, "sigma": m.sigma_,
                    "epsilon": m.epsilon_spent_, "steps": m.steps_,
                    "seconds": time.time() - t}
        k = "r2" if task == "los" else "roc_auc"
        print(f"{key}: {k}={u[k]:.3f} det={f['detection_auc']:.3f} "
              f"prec={f['precision']:.3f} rec={f['recall']:.3f} "
              f"sigma={m.sigma_:.3g} ({time.time()-t:.0f}s)", flush=True)
        json.dump(res, open(out_p, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="los")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--extra", default="{}",
                    help='JSON of DPTabDDPM overrides, e.g. {"lr": 5e-4}')
    a = ap.parse_args()
    main(a.task, a.device, json.loads(a.extra))
