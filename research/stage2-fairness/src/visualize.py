"""
visualize.py
------------
Stage-2 figures from results/metrics.json:

  * fairness to utility trade-off per task: utility (x) vs the headline disparity
    metric (y) for every arm, the plot the stage exists to draw;
  * representation: real vs synthetic subgroup shares (insurance, readmission)
    across arms, showing what fair sampling actually moved.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["TRTR", "stage0", "stage1", "dpbaseline", "fairsample", "fairtrain"]
COLORS = {"TRTR": "k", "stage0": "tab:orange", "stage1": "tab:red",
          "dpbaseline": "tab:gray", "fairsample": "tab:green",
          "fairtrain": "tab:blue"}


def main():
    with open("results/metrics.json") as f:
        m = json.load(f)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # readmission: AUC vs equalized-odds gap (gender)
    a = ax[0]
    for arm in ARMS:
        r = m["readmission"][arm]
        eo = r["fairness"].get("gender", {}).get("eo_diff")
        if eo is None:
            continue
        a.scatter(r["roc_auc"], eo, c=COLORS[arm], s=60, label=arm)
    a.set(xlabel="ROC-AUC (real test)", ylabel="EO diff (gender)",
          title="Readmission, utility vs disparity")
    a.legend(fontsize=8)
    # LOS: R² vs RMSE gap (gender)
    a = ax[1]
    for arm in ARMS:
        r = m["los"][arm]
        gap = r["fairness"].get("gender", {}).get("rmse_gap")
        if gap is None:
            continue
        a.scatter(r["r2"], gap, c=COLORS[arm], s=60, label=arm)
    a.set(xlabel="R² (real test)", ylabel="RMSE gap (gender)",
          title="LOS, utility vs disparity")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/plots/fairness_utility.png", dpi=150)

    # representation: insurance shares per arm (readmission)
    rep_arms = ["dpbaseline", "fairsample", "fairtrain"]
    ref = m["readmission"]["dpbaseline"]["representation"]["insurance"]
    cats = sorted(ref.keys())
    x = np.arange(len(cats))
    width = 0.8 / (len(rep_arms) + 1)
    fig2, a = plt.subplots(figsize=(10, 4.2))
    a.bar(x, [ref[c]["real"] for c in cats], width, label="real",
          color="k", alpha=0.55)
    for i, arm in enumerate(rep_arms, start=1):
        rep = m["readmission"][arm]["representation"]["insurance"]
        a.bar(x + i * width, [rep.get(c, {}).get("dataset", 0.0)
                              for c in cats], width,
              label=arm, color=COLORS[arm], alpha=0.8)
    a.set_xticks(x + width * len(rep_arms) / 2)
    a.set_xticklabels(cats, fontsize=8)
    a.set(ylabel="share", title="Insurance representation, real vs arms")
    a.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig("results/plots/representation_insurance.png", dpi=150)
    print("wrote results/plots/fairness_utility.png and "
          "representation_insurance.png", flush=True)


if __name__ == "__main__":
    main()
