"""Evaluation metrics and result tables."""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)


def compute_metrics(y: np.ndarray, cv: dict, summary: dict, cfg: dict,
                    best_params: dict, best_cv_score: float,
                    threshold: float = 0.5) -> dict:
    """Aggregate out-of-fold predictions into a metrics dictionary."""
    prob = cv["oof_prob"]
    pred = (prob >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    metrics = {
        "cohort": {
            "n_stays": summary["n_stays"],
            "n_patients": summary["n_patients"],
            "n_positive": summary["n_positive"],
            "positive_rate": summary["positive_rate"],
            "n_readmit_within_30d": summary["n_readmit_within_30d"],
            "n_death_within_30d": summary["n_death_within_30d"],
        },
        "cv_roc_auc_mean": round(float(cv["fold_aucs"].mean()), 4),
        "cv_roc_auc_std": round(float(cv["fold_aucs"].std()), 4),
        "cv_roc_auc_folds": [round(float(a), 4) for a in cv["fold_aucs"]],
        "cv_pr_auc_mean": round(float(cv["fold_ap"].mean()), 4),
        "cv_pr_auc_std": round(float(cv["fold_ap"].std()), 4),
        "oof_roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "oof_pr_auc": round(float(average_precision_score(y, prob)), 4),
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "best_params": best_params,
        "tuning_best_cv_roc_auc": round(float(best_cv_score), 4)
                                  if best_cv_score == best_cv_score else None,
        "upstream_reference_auc": cfg["upstream"]["reference_auc"],
    }
    return metrics


def save_metrics(metrics: dict, cfg: dict) -> str:
    path = os.path.join(cfg["paths"]["results_dir"], "metrics.json")
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    return path


def save_results_table(metrics: dict, cfg: dict) -> str:
    """Markdown comparison table vs the upstream AUC benchmark."""
    ref = cfg["upstream"]["reference_auc"]
    ours = metrics["cv_roc_auc_mean"]
    delta = round(ours - ref, 4)
    c = metrics["cohort"]
    path = os.path.join(cfg["paths"]["results_dir"], "results_table.md")
    lines = [
        "# Replication Results vs Upstream Benchmark",
        "",
        "## Headline comparison",
        "",
        "| Study | Database | Model | ROC-AUC | Notes |",
        "|---|---|---|---|---|",
        f"| Blinder (2019), upstream | MIMIC-III | XGBoost, 5-fold CV | "
        f"**{ref:.2f}** | Readmission label only; MICU subset; SAPS-II/SOFA features |",
        f"| This replication | MIMIC-IV v3.1 | XGBoost, 5-fold CV | "
        f"**{ours:.4f} ± {metrics['cv_roc_auc_std']:.4f}** | "
        f"Readmission OR 30-day death; all ICU units; no SAPS-II/SOFA |",
        f"| Difference (ours − upstream) | | | **{delta:+.4f}** | |",
        "",
        "## This replication in detail",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cohort size (ICU stays) | {c['n_stays']:,} |",
        f"| Unique patients | {c['n_patients']:,} |",
        f"| Positive cases (30-day outcome) | {c['n_positive']:,} |",
        f"| Positive rate | {c['positive_rate']:.4f} |",
        f"| &nbsp;&nbsp;of which ICU readmissions | {c['n_readmit_within_30d']:,} |",
        f"| &nbsp;&nbsp;of which deaths within 30 d | {c['n_death_within_30d']:,} |",
        f"| CV ROC-AUC (mean ± std) | {ours:.4f} ± {metrics['cv_roc_auc_std']:.4f} |",
        f"| Out-of-fold ROC-AUC | {metrics['oof_roc_auc']:.4f} |",
        f"| CV PR-AUC (mean ± std) | {metrics['cv_pr_auc_mean']:.4f} ± {metrics['cv_pr_auc_std']:.4f} |",
        f"| Out-of-fold PR-AUC | {metrics['oof_pr_auc']:.4f} |",
        f"| Accuracy @ {metrics['threshold']:.2f} | {metrics['accuracy']:.4f} |",
        f"| Precision @ {metrics['threshold']:.2f} | {metrics['precision']:.4f} |",
        f"| Recall @ {metrics['threshold']:.2f} | {metrics['recall']:.4f} |",
        f"| F1 @ {metrics['threshold']:.2f} | {metrics['f1']:.4f} |",
        "",
        f"Per-fold ROC-AUC: {', '.join(f'{a:.4f}' for a in metrics['cv_roc_auc_folds'])}.",
        "",
        f"Tuned hyperparameters: `{metrics['best_params']}` "
        f"(grid-search best CV ROC-AUC = {metrics['tuning_best_cv_roc_auc']}).",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path
