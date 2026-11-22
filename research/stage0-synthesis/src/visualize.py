"""
visualize.py
------------
Figures for the baseline synthetic-data study:
  1. utility_tstr_vs_trtr.png, the headline TRTR vs TSTR comparison per task
  2. fidelity_los.png, real vs synthetic marginals, LOS features
  3. fidelity_readmission.png, real vs synthetic marginals, readmission features
  4. beddemand_series.png, real vs synthetic daily-count series + histogram
All figures are aggregate; no individual records are shown.
"""
from __future__ import annotations

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import data as D

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
REAL, SYN = "#2c7fb8", "#e6550d"


def _utility(metrics, out):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    los = metrics["los"]
    ax[0].bar(["TRTR", "TSTR"], [los["TRTR"]["r2"], los["TSTR"]["r2"]],
              color=[REAL, SYN]); ax[0].set_title("LOS, R² (higher better)")
    ax[0].set_ylim(0, max(0.25, los["TRTR"]["r2"] * 1.3))
    rd = metrics["readmission"]
    ax[1].bar(["TRTR", "TSTR"], [rd["TRTR"]["roc_auc"], rd["TSTR"]["roc_auc"]],
              color=[REAL, SYN]); ax[1].set_title("Readmission, ROC-AUC")
    ax[1].set_ylim(0.5, 0.85)
    bd = metrics["beddemand"]
    ax[2].bar(["TRTR", "TSTR"], [bd["TRTR"]["mae"], bd["TSTR"]["mae"]],
              color=[REAL, SYN]); ax[2].set_title("Bed demand, MAE (lower better)")
    for a in ax:
        for i, p in enumerate(a.patches):
            a.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2,
                       p.get_height()), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Train-on-Real vs Train-on-Synthetic, evaluated on the real test set",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def _fidelity(real, syn, cols, title, out):
    n = len(cols)
    fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 3))
    if n == 1:
        ax = [ax]
    for a, col in zip(ax, cols):
        if real[col].dtype == object or real[col].nunique() < 12:
            r = real[col].astype(str).value_counts(normalize=True)
            s = syn[col].astype(str).value_counts(normalize=True)
            cats = sorted(set(r.index) | set(s.index))
            x = np.arange(len(cats))
            a.bar(x - 0.2, [r.get(c, 0) for c in cats], 0.4, label="real", color=REAL)
            a.bar(x + 0.2, [s.get(c, 0) for c in cats], 0.4, label="syn", color=SYN)
            a.set_xticks(x); a.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
        else:
            lo, hi = np.nanpercentile(real[col], [1, 99])
            bins = np.linspace(lo, hi, 30)
            a.hist(real[col], bins=bins, density=True, alpha=0.5, label="real", color=REAL)
            a.hist(syn[col], bins=bins, density=True, alpha=0.5, label="syn", color=SYN)
        a.set_title(col, fontsize=9); a.legend(fontsize=7)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def _beddemand(cfg, out):
    real = D.load_beddemand(cfg)["y_train"]
    syn = pd.read_csv(os.path.join(cfg["synthetic_root"],
                                   "beddemand_synthetic.csv"))["count"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(real[:365], color=REAL, lw=0.8, label="real")
    ax[0].plot(syn[:365], color=SYN, lw=0.8, alpha=0.8, label="synthetic")
    ax[0].set_title("Daily ICU-admission count (first year)")
    ax[0].set_xlabel("day"); ax[0].legend(fontsize=8)
    m = int(max(real.max(), syn.max()))
    bins = np.arange(0, m + 2) - 0.5
    ax[1].hist(real, bins=bins, density=True, alpha=0.5, label="real", color=REAL)
    ax[1].hist(syn, bins=bins, density=True, alpha=0.5, label="synthetic", color=SYN)
    ax[1].set_title("Count distribution"); ax[1].set_xlabel("admissions/day")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    os.makedirs("results/plots", exist_ok=True)
    with open("results/metrics.json") as f:
        metrics = json.load(f)
    sroot = cfg["synthetic_root"]

    _utility(metrics, "results/plots/utility_tstr_vs_trtr.png")

    los = D.clean_numeric(D.load_los(cfg)["train"], ["gender", "admission_type", "unitcode"])
    los_syn = pd.read_parquet(os.path.join(sroot, "los_synthetic.parquet"))
    _fidelity(los, los_syn, ["los_days", "age", "creatinine", "gender"],
              "LOS, real vs synthetic marginals", "results/plots/fidelity_los.png")

    rd = D.load_readmission(cfg)
    rd_real = D.clean_numeric(rd["train"], rd["discrete"])
    rd_syn = pd.read_parquet(os.path.join(sroot, "readmission_synthetic.parquet"))
    _fidelity(rd_real, rd_syn, ["age", "los", "diagnosis_count", "label"],
              "Readmission, real vs synthetic marginals",
              "results/plots/fidelity_readmission.png")

    _beddemand(cfg, "results/plots/beddemand_series.png")
    print("wrote 4 figures to results/plots/", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
