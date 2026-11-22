"""
make_plots.py
-------------
Renders the four result figures (PNG, results/plots/) from the backtest
payload. Figure (b) mirrors the upstream signature forecast plot
(plot_forecasts.py in tufts-ml/single-hospital-count-forecasting): observed
history, then the 2.5/50/97.5 percentiles of the sampled forecasts.

All figures show aggregate daily counts only, no patient-level data.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genpoisson import gp_logpmf_table  # noqa: E402

# --- reference palette (light mode) ----------------------------------------
BLUE = "#2a78d6"     # series 1: GP-AR
RED = "#e34948"      # series 6: Poisson-AR
AQUA = "#1baf7a"     # series 2: seasonal naive
YELLOW = "#eda100"   # series 3: moving average
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def load_payload(cfg):
    z = np.load(os.path.join(cfg["derived_dir"], "backtest_payload.npz"),
                allow_pickle=False)
    return z


def gp_quantiles(theta, lam, qs):
    lp = gp_logpmf_table(float(theta), float(lam))
    cdf = np.cumsum(np.exp(lp))
    cdf[-1] = 1.0
    return [int(np.searchsorted(cdf, q)) for q in qs]


def fig_series(cfg, z, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, target, label in [
            (axes[0], "admissions", "Daily ICU admissions"),
            (axes[1], "census", "Daily ICU census (occupancy)")]:
        dates = pd.to_datetime(z[f"{target}__dates"])
        y = z[f"{target}__y"]
        ts = int(z[f"{target}__test_start"])
        roll = pd.Series(y).rolling(28, center=True).mean()
        ax.axvspan(dates[ts], dates[-1], color=GRID, alpha=0.45, zorder=0)
        ax.plot(dates, y, color=BLUE, lw=0.5, alpha=0.35, label="daily count")
        ax.plot(dates, roll, color=BLUE, lw=2, solid_capstyle="round",
                label="28-day rolling mean")
        ax.set_title(f"{label}, MIMIC-IV, de-identified calendar "
                     f"{cfg['window_start'][:4]}-{cfg['window_end'][:4]}")
        ax.set_ylabel("count")
        ax.legend(loc="upper left", ncol=2)
        ax.text(dates[ts + 10], ax.get_ylim()[1] * 0.93, "test year",
                color=SECONDARY, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[1].set_xlabel("de-identified date (per-patient shifted; not real time)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_forecast(cfg, z, target, label, out):
    """Signature plot: history dots + forecast percentiles, upstream-style."""
    dates = pd.to_datetime(z[f"{target}__dates"])
    y = z[f"{target}__y"]
    origin = int(z[f"{target}__sig_origin"])
    F = cfg["horizon"]
    smp = z[f"{target}__sig_samples__gp_ar"]
    smp_po = z[f"{target}__sig_samples__poisson_ar"]

    lo, med, hi = [np.percentile(smp, q, axis=0) for q in (2.5, 50, 97.5)]
    lo50, hi50 = [np.percentile(smp, q, axis=0) for q in (25, 75)]
    lo_po, hi_po = [np.percentile(smp_po, q, axis=0) for q in (2.5, 97.5)]

    n_past = 30
    x_past = np.arange(-n_past, 0)
    x_fut = np.arange(F)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axvline(-0.5, color=BASELINE, lw=1)
    ax.fill_between(x_fut, lo, hi, color=BLUE, alpha=0.14, lw=0,
                    label="GP-AR 95% interval")
    ax.fill_between(x_fut, lo50, hi50, color=BLUE, alpha=0.28, lw=0,
                    label="GP-AR 50% interval")
    ax.plot(x_fut, med, color=BLUE, lw=2, marker="x", ms=6,
            label="GP-AR median")
    ax.plot(x_fut, lo_po, color=RED, lw=1.4, ls="--")
    ax.plot(x_fut, hi_po, color=RED, lw=1.4, ls="--",
            label="Poisson-AR 95% bounds")
    ax.plot(x_past, y[origin - n_past:origin], ".", color=INK, ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1, label="observed")
    ax.plot(x_fut, y[origin:origin + F], ".", color=INK, ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1)

    ticks = [-n_past, 0, F - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(dates[origin + t].date()) for t in ticks])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("count")
    ax.set_xlabel("de-identified date")
    ax.set_title(f"{label}: 14-day-ahead forecast at the final backtest origin")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_gp_vs_poisson(cfg, z, metrics, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, target in enumerate(["admissions", "census"]):
        mt = metrics["targets"][target]["models"]
        # left: per-origin heldout joint loglik is not in payload; use per-origin
        # RB per-day predictive loglik reconstructed from stored quantiles is not
        # possible either, plot the summary metric per model instead as bars,
        # plus one-step loglik.
        ax = axes[row, 0]
        names = ["gp_ar", "poisson_ar"]
        disp = {"gp_ar": ("GP-AR", BLUE), "poisson_ar": ("Poisson-AR", RED)}
        width = 0.32
        xs = np.arange(2)
        joint = [mt[n]["heldout_loglik_per_day_joint"] for n in names]
        ones = [mt[n]["one_step_loglik_test"] for n in names]
        for i, n in enumerate(names):
            lbl, c = disp[n]
            ax.bar(xs[i] - width / 2, joint[i], width, color=c, alpha=0.9,
                   label=None, zorder=3)
            ax.bar(xs[i] + width / 2, ones[i], width, color=c, alpha=0.45,
                   zorder=3)
            ax.text(xs[i] - width / 2, joint[i] - 0.015, f"{joint[i]:.3f}",
                    ha="center", va="top", fontsize=9, color=SECONDARY)
            ax.text(xs[i] + width / 2, ones[i] - 0.015, f"{ones[i]:.3f}",
                    ha="center", va="top", fontsize=9, color=SECONDARY)
        ax.set_xticks(xs)
        ax.set_xticklabels([disp[n][0] for n in names], color=INK)
        ax.set_ylabel("heldout log-likelihood per day")
        ymin = min(joint + ones)
        ax.set_ylim(ymin * 1.25, 0)
        ax.set_title(f"{target}, heldout log-likelihood/day", fontsize=10.5)
        ax.text(0.02, 0.02, "solid: 14-day joint    faded: one-step",
                transform=ax.transAxes, fontsize=8.5, color=SECONDARY,
                va="bottom")
        ax.grid(axis="x", visible=False)

        # right: PIT calibration histograms (one-step, test year)
        ax = axes[row, 1]
        bins = np.linspace(0, 1, 11)
        for n in names:
            lbl, c = disp[n]
            u = z[f"{target}__pit__{n}"]
            h, _ = np.histogram(u, bins=bins)
            ax.stairs(h / len(u), bins, color=c, lw=2, label=lbl)
        ax.axhline(0.1, color=MUTED, lw=1, ls=":")
        ax.text(0.995, 0.102, "uniform", ha="right", va="bottom", fontsize=8,
                color=MUTED)
        ax.set_ylim(0, None)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("relative frequency")
        ax.set_title(f"{target}, one-step PIT calibration", fontsize=10.5)
        ax.text(0.02, 0.98,
                f"MACE  GP {mt['gp_ar']['pit_mace_10bin']:.3f}   "
                f"Poisson {mt['poisson_ar']['pit_mace_10bin']:.3f}",
                transform=ax.transAxes, fontsize=8.5, color=SECONDARY, va="top")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Generalized Poisson vs standard Poisson likelihood "
                 "(same AR mean model)", fontweight="semibold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def fig_dispersion(cfg, z, metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # left: admissions marginal histogram + iid GP and Poisson pmf overlays
    ax = axes[0]
    target = "admissions"
    mfit = json.loads(str(z[f"{target}__marginal_fit_json"]))
    y = z[f"{target}__y"][: int(z[f"{target}__test_start"])]
    ymax = int(y.max()) + 2
    ks = np.arange(ymax + 1)
    freq = np.array([(y == k).mean() for k in ks])
    ax.bar(ks, freq, width=0.82, color=GRID, edgecolor=SURFACE, linewidth=2,
           label="observed frequency", zorder=2)
    from scipy.stats import poisson
    mu = mfit["poisson"]["mu"]
    th, lam = mfit["genpoisson"]["theta"], mfit["genpoisson"]["lambda"]
    gp_pmf = np.exp(gp_logpmf_table(th, lam,
                                    ymax=max(ymax, int(-th / lam) if lam < 0 else ymax)))
    ax.plot(ks, poisson.pmf(ks, mu), "o--", color=RED, lw=1.6, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1,
            label=f"Poisson MLE (μ={mu:.2f})")
    ax.plot(ks, gp_pmf[:ymax + 1], "o-", color=BLUE, lw=1.6, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1,
            label=f"Gen. Poisson MLE (θ={th:.2f}, λ={lam:.3f})")
    s = metrics["targets"][target]["series"]
    ax.set_title(f"Admissions marginal fit,  Var/Mean = "
                 f"{s['dispersion_ratio_var_over_mean']:.2f} (equidispersed)",
                 fontsize=10.5)
    ax.set_xlabel("daily ICU admissions")
    ax.set_ylabel("probability")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)

    # right: census one-step 95% predictive intervals, GP vs Poisson
    ax = axes[1]
    target = "census"
    dates = pd.to_datetime(z[f"{target}__dates"])
    y = z[f"{target}__y"]
    ts = int(z[f"{target}__test_start"])
    n_show = 56
    sl = slice(len(y) - n_show, len(y))
    th_gp = z[f"{target}__onestep_theta__gp_ar"][-n_show:]
    th_po = z[f"{target}__onestep_theta__poisson_ar"][-n_show:]
    lam_gp = float(z[f"{target}__onestep_lam__gp_ar"])
    x = np.arange(n_show)
    q_gp = np.array([gp_quantiles(t, lam_gp, [0.025, 0.975]) for t in th_gp])
    q_po = np.array([gp_quantiles(t, 0.0, [0.025, 0.975]) for t in th_po])
    ax.fill_between(x, q_po[:, 0], q_po[:, 1], color=RED, alpha=0.12, lw=0,
                    label="Poisson-AR 95% one-step interval")
    ax.fill_between(x, q_gp[:, 0], q_gp[:, 1], color=BLUE, alpha=0.22, lw=0,
                    label="GP-AR 95% one-step interval")
    ax.plot(x, y[sl], ".", color=INK, ms=6, markeredgecolor=SURFACE,
            markeredgewidth=1, label="observed census")
    mt = metrics["targets"][target]
    lam_txt = mt["fixed_fit_params"]["gp_ar"]["lambda"]
    ax.set_title(f"Census conditional,  Pearson "
                 f"{mt['conditional_dispersion_pearson_poisson_ar']:.2f}, "
                 f"GP λ={lam_txt:.2f} (under-dispersed)", fontsize=10.5)
    ticks = [0, n_show // 2, n_show - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(dates[len(y) - n_show + t].date()) for t in ticks])
    ax.set_xlabel("de-identified date (last 8 weeks of test year)")
    ax.set_ylabel("census count")
    ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Why the Generalized Poisson: dispersion diagnostics",
                 fontweight="semibold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out)
    plt.close(fig)


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(cfg["results_dir"], "metrics.json")) as f:
        metrics = json.load(f)
    z = load_payload(cfg)
    plot_dir = os.path.join(cfg["results_dir"], "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig_series(cfg, z, os.path.join(plot_dir, "series_daily_counts.png"))
    fig_forecast(cfg, z, "admissions", "Daily ICU admissions",
                 os.path.join(plot_dir, "forecast_admissions.png"))
    fig_forecast(cfg, z, "census", "Daily ICU census",
                 os.path.join(plot_dir, "forecast_census.png"))
    fig_gp_vs_poisson(cfg, z, metrics,
                      os.path.join(plot_dir, "genpoisson_vs_poisson.png"))
    fig_dispersion(cfg, z, metrics,
                   os.path.join(plot_dir, "dispersion_diagnostics.png"))
    print(f"wrote 5 figures to {plot_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
