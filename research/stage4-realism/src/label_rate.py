"""
label_rate.py
-------------
The positive-label rate of every readmission release, against the real
training rate (about 13.9%).

Added after the main run, because it explains a result: ranking metrics
(ROC-AUC) ignore a distorted label marginal, but anything thresholded does
not. The downstream model sets its class weight from the release's own label
rate, so a release with too many or too few positives yields a classifier
that predicts one class at the 0.5 threshold, and threshold-based fairness
metrics (demographic parity, equalised odds) become vacuous.

Writes results/label_rate.json.
"""
from __future__ import annotations

import json
import os

import jobs as J
import realism_data as RD
import settings
from realism_posthoc import dataset

RES = os.environ.get("SYNTHEHR_RESULTS_DIR", os.path.join(settings.HERE, "results"))


def main():
    cfg = settings.load()
    out = {}
    for seed in cfg["seeds"]:
        c = dict(cfg); c["seed"] = seed
        d = RD.load_with_groups("readmission", c)
        for arm in J.ALL_ARMS:
            if not os.path.exists(J.result_path("readmission", seed, arm)):
                continue
            syn = dataset(c, "readmission", seed, arm, d)
            if syn is None:
                continue
            out[f"readmission/{seed}/{arm}"] = float(
                (syn["label"].astype(str) == "1").mean())
    with open(os.path.join(RES, "label_rate.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"{len(out)} entries")


if __name__ == "__main__":
    main()
