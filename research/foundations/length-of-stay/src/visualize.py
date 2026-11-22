"""
visualize.py, Aggregate diagnostic plots for the LOS replication.

Produces five PNGs under results/plots/ (all aggregate/distribution views,
no individual-patient tables, per PhysioNet DUA):

  a) predicted_vs_actual.png, predicted vs observed ICU LOS (hexbin density
                                 + 45-degree reference line)
  b) residual_distribution.png, histogram of (observed - predicted) residuals
  c) feature_importance.png, RF base-learner importances (top features)
  d) los_distribution.png, histogram of observed truncated LOS
  e) slos_by_unit.png, per-ICU-unit observed vs predicted totals
                                 (SLOS funnel-style bar/scatter)

Reads : data/derived/test_predictions.parquet, data/derived/feature_importances.parquet
Writes: results/plots/*.png
Usage : python src/visualize.py   (or: make eval)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_config

CLIP = 21.0
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3})


def main():
    config = load_config()
    d = config["derived_dir"]
    plots = f"{config['results_dir']}/plots"
    os.makedirs(plots, exist_ok=True)

    pred = pd.read_parquet(f"{d}/test_predictions.parquet")
    imp = pd.read_parquet(f"{d}/feature_importances.parquet")
    obs, prd = pred["observed"].values, pred["predicted"].values

    # (a) predicted vs actual, hexbin density (too many points for a scatter)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    # bins with fewer than 10 stays are not drawn (same small-cell rule as
    # the per-unit table)
    hb = ax.hexbin(obs, prd, gridsize=45, cmap="viridis", mincnt=10,
                   bins="log")
    ax.plot([0, CLIP], [0, CLIP], "r--", lw=1.5, label="perfect (y = x)")
    ax.set_xlim(0, CLIP); ax.set_ylim(0, CLIP)
    ax.set_xlabel("Observed ICU LOS (days, truncated at 21)")
    ax.set_ylabel("Predicted ICU LOS (days)")
    ax.set_title("Predicted vs Observed ICU LOS (test set)")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout(); fig.savefig(f"{plots}/predicted_vs_actual.png"); plt.close(fig)

    # (b) residual distribution
    resid = obs - prd
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(resid, bins=60, color="#4c72b0", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.set_xlabel("Residual = observed - predicted (days)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution (mean={resid.mean():.2f}, "
                 f"std={resid.std():.2f} days)")
    fig.tight_layout(); fig.savefig(f"{plots}/residual_distribution.png"); plt.close(fig)

    # (c) feature importance (top 20)
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(top["feature"], top["importance"], color="#55a868")
    ax.set_xlabel("Random-forest impurity importance")
    ax.set_title("Base random-forest feature importance (top 20)")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(f"{plots}/feature_importance.png"); plt.close(fig)

    # (d) LOS distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(obs, bins=np.arange(0, CLIP + 1, 0.5), color="#c44e52",
            edgecolor="white")
    ax.set_xlabel("Observed ICU LOS (days, truncated at 21)")
    ax.set_ylabel("Count")
    ax.set_title("ICU LOS distribution (test set)")
    fig.tight_layout(); fig.savefig(f"{plots}/los_distribution.png"); plt.close(fig)

    # (e) per-ICU-unit observed vs predicted totals (SLOS view)
    unit = (pred.groupby("unitcode")
            .agg(admissions=("observed", "size"),
                 obs_total=("observed", "sum"),
                 pred_total=("predicted", "sum"))
            .reset_index())
    unit["slos"] = unit["obs_total"] / unit["pred_total"]
    from evaluate import suppress_small_units   # same small-cell rule
    unit = suppress_small_units(unit).sort_values("admissions",
                                                  ascending=False)
    # Shorten long care-unit names for the axis.
    unit["short"] = unit["unitcode"].str.replace(r"\s*\(.*\)", "", regex=True).str[:22]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    m = max(unit["obs_total"].max(), unit["pred_total"].max()) * 1.05
    ax.scatter(unit["pred_total"], unit["obs_total"],
               s=np.sqrt(unit["admissions"]) * 6, c="#4c72b0", alpha=0.8,
               edgecolor="k")
    ax.plot([0, m], [0, m], "r--", lw=1.3, label="SLOS = 1")
    for _, r in unit.iterrows():
        ax.annotate(r["short"], (r["pred_total"], r["obs_total"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("Predicted total LOS per unit (days)")
    ax.set_ylabel("Observed total LOS per unit (days)")
    ax.set_title("Unit-level calibration (marker area ~ admissions)")
    ax.legend()

    ax = axes[1]
    order = unit.sort_values("slos")
    colors = ["#c44e52" if s > 1 else "#55a868" for s in order["slos"]]
    ax.barh(order["short"], order["slos"], color=colors)
    ax.axvline(1.0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("SLOS = observed / predicted")
    ax.set_title("Standardized LOS (SLOS) by ICU unit")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(f"{plots}/slos_by_unit.png"); plt.close(fig)

    print(f"Saved 5 plots to {plots}/")
    for p in ["predicted_vs_actual", "residual_distribution", "feature_importance",
              "los_distribution", "slos_by_unit"]:
        print(f"  {p}.png")


if __name__ == "__main__":
    main()
