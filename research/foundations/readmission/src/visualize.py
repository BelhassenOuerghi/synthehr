"""Figures for the replication report (all aggregate, no patient-level rows)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _fig_path(cfg: dict, name: str) -> str:
    return os.path.join(cfg["paths"]["figures_dir"], name)


def plot_roc_cv(cv: dict, cfg: dict) -> str:
    """ROC curve with per-fold lines, mean and ±1 std band (mirrors upstream)."""
    mean_fpr = cv["mean_fpr"]
    tprs = cv["tprs"]
    fold_aucs = cv["fold_aucs"]
    mean_tpr = cv["mean_tpr"]
    mean_auc = cv["mean_auc"]

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (tpr, a) in enumerate(zip(tprs, fold_aucs)):
        ax.plot(mean_fpr, tpr, lw=1.2, alpha=0.6,
                label=f"Fold {i} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.5, label="Chance")

    std_tpr = tprs.std(axis=0)
    upper = np.minimum(mean_tpr + std_tpr, 1)
    lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, lower, upper, color="tab:blue", alpha=0.2,
                    label="±1 std. dev.")
    ax.plot(mean_fpr, mean_tpr, color="navy", lw=3,
            label=f"Mean ROC (AUC = {mean_auc:.3f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("30-day ICU readmission, ROC (5-fold CV, MIMIC-IV)")
    ax.legend(loc="lower right", fontsize=10)
    path = _fig_path(cfg, "roc_cv.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(imp: pd.Series, cfg: dict, top_n: int = 20) -> str:
    top = imp.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top.index, top.values, color="teal")
    ax.set_xlabel("Importance (XGBoost gain)")
    ax.set_title(f"Top {top_n} features by gain")
    path = _fig_path(cfg, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_label_balance(summary: dict, cfg: dict) -> str:
    n_pos = summary["n_positive"]
    n_neg = summary["n_stays"] - n_pos
    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(["Not readmitted (0)", "Readmitted / died ≤30d (1)"],
                  [n_neg, n_pos], color=["tab:blue", "tab:red"])
    for b, v in zip(bars, [n_neg, n_pos]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/summary['n_stays']:.1%})",
                ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("ICU stays")
    ax.set_title("Class balance (30-day outcome)")
    ax.set_ylim(0, summary["n_stays"] * 1.08)
    path = _fig_path(cfg, "label_balance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_top_feature_kde(df: pd.DataFrame, imp: pd.Series, cfg: dict,
                         top_n: int = 6) -> str:
    """KDE of the top numeric features, split by class (upstream-style)."""
    # Restrict to numeric, non-dummy features present in the raw frame.
    numeric = [f for f in imp.index
               if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
               and df[f].nunique(dropna=True) > 10]
    feats = numeric[:top_n]
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        d = df[[f, "label"]].dropna()
        # clip to 1-99th percentile for readable axes
        lo, hi = d[f].quantile([0.01, 0.99])
        d = d[(d[f] >= lo) & (d[f] <= hi)]
        for lbl, colour in [(0, "tab:blue"), (1, "tab:red")]:
            sns.kdeplot(d[d.label == lbl][f], ax=ax, fill=True, alpha=0.35,
                        color=colour, label=("readmit/die" if lbl else "no event"))
        ax.set_title(f, fontsize=13)
        ax.set_xlabel("")
        ax.legend(fontsize=9)
    for ax in axes[len(feats):]:
        ax.set_visible(False)
    fig.suptitle("Top features by class (KDE)", y=1.01, fontsize=16)
    path = _fig_path(cfg, "top_features_kde.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all_figures(df: pd.DataFrame, cv: dict, imp: pd.Series,
                     summary: dict, cfg: dict) -> list[str]:
    return [
        plot_roc_cv(cv, cfg),
        plot_feature_importance(imp, cfg),
        plot_label_balance(summary, cfg),
        plot_top_feature_kde(df, imp, cfg),
    ]
