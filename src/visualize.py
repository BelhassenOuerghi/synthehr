"""
visualize.py
------------
Figures for Pillar 4, drawn from results/metrics_ci.json.

  ladder.png       Utility of every rung, with 95% intervals: real data, the
                   two generators without privacy, and the two under DP.
                   The question the pillar exists to answer, in one picture.
  realism.png      Realism measured directly: detection AUC (lower is more
                   realistic) and precision/recall, against the real-vs-real
                   ceiling.
  ensembling.png   Every individual seed and replicate for single DP runs and
                   for pooled releases, so the spread is visible, not summarised.
  ordering.png     Denoiser capacity against utility, with and without DP: the
                   capacity x privacy interaction behind the ordering argument.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import settings  # noqa: E402

RES = os.environ.get("ENGD_RESULTS_DIR",
                     os.path.join(settings.HERE, "results"))
OUT = os.path.join(RES, "plots")
KEY = {"los": "r2", "readmission": "roc_auc"}
TITLE = {"los": "Length of stay (R²)", "readmission": "30-day readmission "
         "(ROC-AUC)"}
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C = {"real": "#24292f", "ctgan": "#0969da", "ddpm": "#1a7f37",
     "dpctgan": "#cf222e", "dpddpm": "#8250df"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def _m(T, arm, k):
    a = T["arms"].get(arm)
    if not a:
        return None
    return (a["utility"].get(k) or a["fidelity"].get(k)
            or a["mia"].get(k))


def fig_ladder(agg):
    rungs = [("TRTR", "Real data", C["real"]),
             ("ctgan", "CTGAN", C["ctgan"]),
             ("ddpm", "TabDDPM", C["ddpm"]),
             ("dpctgan", "DP-CTGAN\nε=10", C["dpctgan"]),
             ("dpddpm", "DP-TabDDPM\nε=10", C["dpddpm"]),
             ("dpddpm_patient", "DP-TabDDPM\npatient-level", C["dpddpm"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, task in zip(axes, ("readmission", "los")):
        T = agg["tasks"].get(task)
        if not T:
            continue
        xs, ys, es, cs, ls = [], [], [], [], []
        for i, (arm, lab, col) in enumerate(rungs):
            c = _m(T, arm, KEY[task])
            if not c:
                continue
            xs.append(i); ys.append(c["mean"]); es.append(c["half"])
            cs.append(col); ls.append(lab)
            ax.scatter([i] * len(c["values"]), c["values"], s=12,
                       color=col, alpha=0.35, zorder=2, linewidths=0)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, capsize=4,
                    lw=1.2, zorder=3)
        ax.scatter(xs, ys, s=46, c=cs, zorder=4, edgecolors="white",
                   linewidths=1)
        ax.set_xticks(xs); ax.set_xticklabels(ls, fontsize=8.5, color=INK)
        ax.set_title(TITLE[task], fontsize=11, color=INK, loc="left")
        if task == "los":
            lo = min(ys) - max(es) - 0.05
            ax.set_ylim(max(lo, -1.0), None)
            ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        else:
            ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        _style(ax)
    fig.suptitle("What realism recovers: utility of each rung "
                 "(trained on synthetic, tested on real; mean ± 95% CI, "
                 "dots are seeds)", fontsize=10.5, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ladder.png"), dpi=160)
    plt.close(fig)


def fig_realism(agg):
    arms = [("TRTR", "Real (held out)"), ("ctgan", "CTGAN"),
            ("ddpm", "TabDDPM"), ("dpctgan", "DP-CTGAN"),
            ("dpddpm", "DP-TabDDPM")]
    cols = [C["real"], C["ctgan"], C["ddpm"], C["dpctgan"], C["dpddpm"]]
    metrics = [("detection_auc", "Detection AUC (↓, 0.5 = indistinguishable)"),
               ("precision", "Precision (↑, plausibility)"),
               ("recall", "Recall (↑, coverage)")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for r, task in enumerate(("readmission", "los")):
        T = agg["tasks"].get(task)
        if not T:
            continue
        for c, (k, lab) in enumerate(metrics):
            ax = axes[r, c]
            vals = [T["arms"].get(a, {}).get("fidelity", {}).get(k)
                    for a, _ in arms]
            y = [v["mean"] if v else np.nan for v in vals]
            e = [v["half"] if v else 0 for v in vals]
            ax.bar(range(len(arms)), y, yerr=e, color=cols, capsize=3,
                   error_kw={"ecolor": INK, "lw": 1})
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([a[1] for a in arms], fontsize=8, rotation=20,
                               color=INK)
            ax.set_ylim(0, 1.05)
            if k == "detection_auc":
                ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
            ax.set_title(f"{task}: {lab}", fontsize=9, color=INK, loc="left")
            _style(ax)
    fig.suptitle("Realism measured directly, against real rows the generator "
                 "never saw", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "realism.png"), dpi=160)
    plt.close(fig)


def fig_ensembling(agg):
    fams = [("dpctgan", "DP-CTGAN", C["dpctgan"]),
            ("dpddpm", "DP-TabDDPM", C["dpddpm"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, task in zip(axes, ("readmission", "los")):
        T = agg["tasks"].get(task)
        if not T:
            continue
        x, ticks = 0, []
        for base, lab, col in fams:
            def eps(a):
                p = T["arms"].get(a, {}).get("privacy", {}).get("epsilon")
                return f"ε={p['mean']:.0f}" if p else "ε=?"
            groups = [("1 run\nε=10", [base] + [f"{base}_r{i}"
                                              for i in (1, 2, 3)]),
                      (f"pool of 4\n{eps(base + '_ens4')}",
                       [f"{base}_ens4"]),
                      (f"pool of 4\n{eps(base + '_ens4split')} split",
                       [f"{base}_ens4split"])]
            for glab, members in groups:
                v = []
                for a in members:
                    c = _m(T, a, KEY[task])
                    if c:
                        v += c["values"]
                if v:
                    jit = np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 \
                        else [0]
                    ax.scatter(x + np.asarray(jit), v, s=16, color=col,
                               alpha=0.75, linewidths=0)
                    ax.hlines(np.mean(v), x - 0.28, x + 0.28, color=INK,
                              lw=1.4)
                ticks.append((x, glab))
                x += 1
            ax.text(x - 2, 1.02, lab, transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=9, color=col)
            x += 0.6
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], fontsize=7.5, color=INK)
        ax.set_title(TITLE[task], fontsize=11, color=INK, loc="left")
        _style(ax)
    fig.suptitle("Does pooling independently trained DP generators buy back "
                 "predictability? (every dot is one release; bar = mean)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ensembling.png"), dpi=160)
    plt.close(fig)


def fig_ordering(agg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    caps = ["S", "M", "L"]
    for ax, task in zip(axes, ("readmission", "los")):
        T = agg["tasks"].get(task)
        if not T:
            continue
        for pre, lab, col in (("ddpm", "without DP", C["ddpm"]),
                              ("dpddpm", "with DP (ε=10)", C["dpddpm"])):
            arms = [f"{pre}_S", pre, f"{pre}_L"]
            cs = [_m(T, a, KEY[task]) for a in arms]
            if any(c is None for c in cs):
                continue
            ax.errorbar(range(3), [c["mean"] for c in cs],
                        yerr=[c["half"] for c in cs], marker="o", color=col,
                        capsize=4, lw=1.6, label=lab)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["S\n2×128", "M\n3×256", "L\n4×512"], color=INK)
        ax.set_title(TITLE[task], fontsize=11, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8.5)
        _style(ax)
    fig.suptitle("Do realism and privacy commute? Denoiser capacity against "
                 "utility, with and without DP", fontsize=10.5, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ordering.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    agg = json.load(open(os.path.join(RES, "metrics_ci.json")))
    for fn in (fig_ladder, fig_realism, fig_ensembling, fig_ordering):
        fn(agg)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
