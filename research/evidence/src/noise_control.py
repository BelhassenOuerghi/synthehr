"""
noise_control.py
----------------
The **negative control** for the multi-seed analysis.

A paired significance test is only trustworthy if a comparison that *should*
show nothing does show nothing. `multiseed.py` intended `stage1` vs
`dpbaseline` to be that control, but in that driver both arms are the same
class with the same seed, so they are bit-identical (Δ = 0.000 exactly). That
identity is a useful *determinism/reproducibility* check, but it is not a noise
control: it compares a model to itself.

This script builds the real thing: for each seed, refit the DP generator with
**identical recipe, identical train/test split, but an independent generator
pipeline seed** (model seed = data seed + 1000). Because the harness draws the
30,000-row fitting subsample with that same seed, the replicate differs from
`stage1` in its initialisation AND its fitting rows: the spread is the noise of
re-running the pipeline, the run-to-run variation that any between-arm
difference must exceed to be a finding. Stage 2 stumbled on this quantity
accidentally (0.05 ROC-AUC / 0.10 R²); here it is measured on purpose.

Writes `results/seeds/noise_<seed>.json` (resumable, skips completed seeds).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import yaml

import data as D
from multiseed import HERE, SCORERS, _cols, _restore, gen_dp

warnings.filterwarnings("ignore")


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sdir = os.path.join(HERE, "results", "seeds")
    os.makedirs(sdir, exist_ok=True)

    for seed in cfg["seeds"]:
        path = os.path.join(sdir, f"noise_{seed}.json")
        if os.path.exists(path):
            print(f"noise control seed {seed}: done, skipping", flush=True)
            continue
        print(f"\n=== NOISE CONTROL, data seed {seed} "
              f"(model seed {seed + 1000}) ===", flush=True)
        t = time.time()
        out = {"data_seed": seed, "model_seed": seed + 1000, "tasks": {}}

        for loader, name in ((D.load_los, "los"),
                             (D.load_readmission, "readmission")):
            dcfg = dict(cfg); dcfg["seed"] = seed          # split: unchanged
            ds = loader(dcfg)
            tgt = ds["target"]
            sens = cfg["fairness"]["sensitive"][name]
            train = D.clean_numeric(ds["train"], ds["discrete"])
            test = D.clean_numeric(ds["test"], ds["discrete"])
            disc, num = _cols(train, tgt, ds["discrete"])

            mcfg = dict(cfg); mcfg["seed"] = seed + 1000   # model: independent
            m = gen_dp(ds, mcfg, train, [], f"{name}-replicate")
            syn = _restore(m.sample(len(train)), train, ds["discrete"])
            # score with the DOWNSTREAM seed held at the data seed, so only the
            # generator's randomness differs from the stage1 arm
            out["tasks"][name] = {
                "dprepeat": SCORERS[name](syn, test, tgt, disc, num, seed, sens),
                "epsilon": m.epsilon_spent_,
            }

        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"=== noise control seed {seed} complete in "
              f"{(time.time()-t)/60:.0f} min ===", flush=True)
    print("\nnoise control complete.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    main(ap.parse_args().config)
