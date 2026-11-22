"""End-to-end pipeline: extract -> prepare -> tune -> CV -> evaluate -> visualise.

Usage:
    python -m src.run_pipeline            # full run (re-extracts if no cache)
    python -m src.run_pipeline --use-cache  # reuse cached feature parquet
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .config import load_config
from .features import build_feature_frame
from .train import prepare_xy, tune, cross_validate, final_fit
from .evaluate import compute_metrics, save_metrics, save_results_table
from .visualize import make_all_figures


def get_feature_frame(cfg: dict, use_cache: bool) -> tuple[pd.DataFrame, dict]:
    cache = os.path.join(cfg["paths"]["work_dir"], "feature_frame.parquet")
    summ_cache = os.path.join(cfg["paths"]["work_dir"], "cohort_summary.json")
    if use_cache and os.path.exists(cache) and os.path.exists(summ_cache):
        print(f"[extract] using cached frame: {cache}")
        df = pd.read_parquet(cache)
        with open(summ_cache) as fh:
            summary = json.load(fh)
        return df, summary
    print("[extract] building cohort + features from MIMIC-IV (DuckDB)...")
    df, summary = build_feature_frame(cfg)
    df.to_parquet(cache, index=False)
    with open(summ_cache, "w") as fh:
        json.dump(summary, fh, indent=2)
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse cached feature parquet if present")
    args = ap.parse_args()

    cfg = load_config()
    df, summary = get_feature_frame(cfg, args.use_cache)
    print(f"[cohort] {summary}")

    X, y, feat_names = prepare_xy(df)
    print(f"[prepare] X={X.shape}, positives={int(y.sum())} ({y.mean():.3f})")

    print("[tune] grid search...")
    best_params, best_score = tune(X, y, cfg)
    print(f"[tune] best_params={best_params} best_cv_auc={best_score:.4f}")

    print("[cv] 5-fold stratified cross-validation...")
    cv = cross_validate(X, y, cfg, best_params)
    print(f"[cv] ROC-AUC {cv['fold_aucs'].mean():.4f} ± {cv['fold_aucs'].std():.4f}")

    print("[fit] final model for feature importance...")
    _, imp = final_fit(X, y, cfg, best_params)
    imp.to_csv(os.path.join(cfg["paths"]["results_dir"],
                            "feature_importance.csv"), header=["gain"])

    metrics = compute_metrics(y, cv, summary, cfg, best_params, best_score)
    mpath = save_metrics(metrics, cfg)
    tpath = save_results_table(metrics, cfg)
    print(f"[eval] wrote {mpath} and {tpath}")

    figs = make_all_figures(df, cv, imp, summary, cfg)
    print("[viz] figures:", [os.path.basename(f) for f in figs])
    print("[done] cv_roc_auc_mean =", metrics["cv_roc_auc_mean"])


if __name__ == "__main__":
    main()
