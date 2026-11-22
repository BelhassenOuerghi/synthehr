"""
visualize.py
------------
Stage-1 figures, from results/metrics.json:

  * privacy to utility frontier per task: downstream metric vs ε, with the TRTR
    ceiling and the Stage-0 (non-private, ε = ∞) TSTR as reference lines;
  * privacy leakage: membership-inference attack AUC vs ε, with the 0.5
    chance line every DP release should hug.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open("results/metrics.json") as f:
        m = json.load(f)
    eps = sorted(m["los"]["TSTR_by_epsilon"].keys(), key=float)
    x = [float(e) for e in eps]
    s0 = m.get("stage0_reference") or {}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # -- LOS R² ---------------------------------------------------------------
    a = ax[0]
    a.plot(x, [m["los"]["TSTR_by_epsilon"][e]["r2"] for e in eps],
           "o-", label="TSTR(ε)")
    a.axhline(m["los"]["TRTR"]["r2"], ls="--", c="k", label="TRTR ceiling")
    if s0.get("los"):
        a.axhline(s0["los"]["TSTR"]["r2"], ls=":", c="tab:orange",
                  label="Stage-0 TSTR (ε=∞)")
    a.set(xscale="log", xlabel="ε (log)", ylabel="R²",
          title="LOS, utility vs privacy budget")
    a.legend(fontsize=8)

    # -- readmission ROC-AUC ---------------------------------------------------
    a = ax[1]
    a.plot(x, [m["readmission"]["TSTR_by_epsilon"][e]["roc_auc"] for e in eps],
           "o-", label="TSTR(ε)")
    a.axhline(m["readmission"]["TRTR"]["roc_auc"], ls="--", c="k",
              label="TRTR ceiling")
    if s0.get("readmission"):
        a.axhline(s0["readmission"]["TSTR"]["roc_auc"], ls=":",
                  c="tab:orange", label="Stage-0 TSTR (ε=∞)")
    a.axhline(0.5, ls=":", c="gray", lw=0.8)
    a.set(xscale="log", xlabel="ε (log)", ylabel="ROC-AUC",
          title="Readmission, utility vs privacy budget")
    a.legend(fontsize=8)

    # -- MIA attack AUC ---------------------------------------------------------
    a = ax[2]
    for task, style in (("los", "o-"), ("readmission", "s-")):
        a.plot(x, [m[task]["MIA_by_epsilon"][e]["attack_auc_knn_classifier"]
                   for e in eps], style, label=f"{task} (kNN attack)")
        a.plot(x, [m[task]["MIA_by_epsilon"][e]["attack_auc_dcr"]
                   for e in eps], style, alpha=0.4,
               label=f"{task} (DCR attack)")
    a.axhline(0.5, ls="--", c="k", label="chance (no leakage)")
    a.set(xscale="log", xlabel="ε (log)", ylabel="attack AUC",
          title="Membership inference vs privacy budget")
    a.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("results/plots/privacy_utility_frontier.png", dpi=150)
    print("wrote results/plots/privacy_utility_frontier.png", flush=True)


if __name__ == "__main__":
    main()
