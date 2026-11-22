"""
visualize.py
------------
Figures for the evidence-hardening repository. These are the programme's headline
results, and the third one is the most important figure in the thesis.

  1. utility_with_ci.png    Utility per arm with 95% confidence intervals. Shows the
                            cost of synthesis and the cost of privacy, and shows that
                            the DP arms have error bars wide enough to swallow the
                            differences between them.
  2. instability.png        Run-to-run standard deviation, non-private vs DP, on three
                            independent measurements (readmission utility, LoS utility,
                            subgroup representation). This is the "DP destroys
                            predictability" finding.
  3. noise_floor.png        The measured noise floor against the size of the effects the
                            programme claimed to measure. The interventions are smaller
                            than the noise. This is the figure that killed two of our own
                            findings, and it is the one to put in front of an examiner.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "plots")

ARMS = ["TRTR", "stage0", "stage1", "fairsample", "fairtrain"]
LABEL = {"TRTR": "Real data\n(ceiling)", "stage0": "Synthetic\n(no privacy)",
         "stage1": "Synthetic\n+ DP", "fairsample": "+ fair\nsampling",
         "fairtrain": "+ fair\ntraining"}
COLOR = {"TRTR": "#222222", "stage0": "#1f77b4", "stage1": "#d62728",
         "fairsample": "#2ca02c", "fairtrain": "#9467bd"}


def fig_utility(m):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (task, met, name) in zip(axes, (
            ("readmission", "roc_auc", "30-day readmission (ROC-AUC)"),
            ("los", "r2", "Length of stay (R²)"))):
        xs, ys, es, cs = [], [], [], []
        for i, arm in enumerate(ARMS):
            d = m["tasks"][task][arm][met]
            xs.append(i); ys.append(d["mean"]); es.append(d["ci95_half_width"])
            cs.append(COLOR[arm])
        ax.errorbar(xs, ys, yerr=es, fmt="o", capsize=5, markersize=7,
                    ecolor="#888888", elinewidth=1.5, zorder=3, linestyle="none")
        for x, y, c in zip(xs, ys, cs):
            ax.plot(x, y, "o", color=c, markersize=7, zorder=4)
        ax.axhline(ys[0], ls="--", lw=0.9, c="#222222", alpha=0.5, zorder=1)
        if met == "roc_auc":
            ax.axhline(0.5, ls=":", lw=0.9, c="grey", zorder=1)
            ax.text(4.3, 0.505, "chance", fontsize=7, color="grey", va="bottom")
        else:
            ax.axhline(0.0, ls=":", lw=0.9, c="grey", zorder=1)
            ax.text(4.3, 0.02, "predicting the mean", fontsize=7, color="grey")
        ax.set_xticks(xs); ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Utility of each arm, 5 seeds, 95% confidence intervals\n"
                 "(patient-disjoint splits; the real-data ceiling is dashed)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT, "utility_with_ci.png"), dpi=150)


def fig_instability(m):
    labels = ["Readmission\nROC-AUC", "Length of stay\nR²",
              "Medicare share\nin the cohort"]
    non_private, dp = [], []
    for task, met in (("readmission", "roc_auc"), ("los", "r2")):
        non_private.append(np.std(m["tasks"][task]["stage0"][met]["values"], ddof=1))
        dp.append(np.std(m["tasks"][task]["stage1"][met]["values"], ddof=1))
    non_private.append(0.080); dp.append(0.223)      # representation (from per-seed audit)

    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w / 2, non_private, w, label="Without privacy", color="#1f77b4")
    ax.bar(x + w / 2, dp, w, label="With differential privacy", color="#d62728")
    for i, (a, b) in enumerate(zip(non_private, dp)):
        ax.text(i + w / 2, b, f"  {b/max(a,1e-9):.1f}x", va="bottom", ha="center",
                fontsize=10, fontweight="bold", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Run-to-run standard deviation\n(same recipe, different seed)")
    ax.set_title("Differential privacy does not merely cost accuracy.\n"
                 "It destroys predictability, on every axis measured.", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "instability.png"), dpi=150)


def fig_noise_floor(m):
    """Effect sizes against the measured noise floor.

    Colour encodes the paired significance test across seeds, NOT position relative
    to the line. The two are different questions and conflating them would be wrong:
    the noise floor is measured between *DP-trained* replicates, which are unstable,
    so it governs comparisons that involve a DP arm. A comparison between two stable
    arms (real data vs non-private synthesis) can be resolved at a smaller effect
    size, and is.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # (label, effect, p-value) from results/metrics_ci.json comparisons
    spec = [
        ("readmission", 0.151, "ROC-AUC", "30-day readmission",
         [("cost of synthesis", 0.089, 0.0002), ("cost of privacy", 0.171, 0.0053),
          ("fair sampling", 0.035, 0.2683), ("fair training", 0.041, 0.2360)]),
        ("los", 3.08, "R²", "Length of stay",
         [("cost of synthesis", 0.176, 0.0042), ("cost of privacy", 0.681, 0.2789),
          ("fair sampling", 0.056, 0.5291), ("fair training", 0.745, 0.5178)]),
    ]
    for ax, (task, floor, unit, title, effects) in zip(axes, spec):
        names = [e[0] for e in effects][::-1]
        vals = [e[1] for e in effects][::-1]
        ps = [e[2] for e in effects][::-1]
        cols = ["#2ca02c" if p < 0.05 else "#bbbbbb" for p in ps]
        ax.barh(names, vals, color=cols, zorder=3)
        for i, (v, p) in enumerate(zip(vals, ps)):
            ax.text(v, i, f"  p={p:.3f}", va="center", fontsize=8,
                    color="#2ca02c" if p < 0.05 else "#777777")
        ax.axvline(floor, color="#d62728", lw=2, zorder=4)
        ax.text(floor, -0.55, f"noise floor {floor:.3f}", color="#d62728",
                fontsize=8.5, fontweight="bold", ha="center")
        ax.set_xlim(0, max(max(vals), floor) * 1.35)
        ax.set_xlabel(f"effect size ({unit})")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2ca02c"),
               plt.Rectangle((0, 0), 1, 1, color="#bbbbbb")]
    axes[0].legend(handles, ["survives (p < 0.05)", "within noise"],
                   fontsize=8, loc="lower right")
    fig.suptitle(
        "Effect sizes against the measured noise floor (5 seeds).\n"
        "The fairness interventions are 3 to 4 times smaller than the noise they must "
        "be measured against, so they cannot be resolved at all.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT, "noise_floor.png"), dpi=150)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "results", "metrics_ci.json")) as f:
        m = json.load(f)
    fig_utility(m)
    fig_instability(m)
    fig_noise_floor(m)
    print("wrote 3 figures to results/plots/", flush=True)


if __name__ == "__main__":
    main()
