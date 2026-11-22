"""
evaluate.py, Test-set metrics, SLOS computation, and result tables.

Computes the metrics reported by the upstream project on the held-out,
patient-level test split:

  Patient-level:
    - RMSE, MAE, R^2

  Unit-level (SLOS = Standardized LOS):
    - Per ICU unit (UnitCode = first_careunit): observed & predicted LOS
      totals and SLOS = sum(observed) / sum(predicted)
    - Overall SLOS
    - Unit-level R^2 on (predicted total vs observed total) across units

Predictions are clipped to [0, 21] days before SLOS aggregation, matching the
upstream Testing.R behaviour.

Reads : models/stacking_model.joblib, data/derived/{cohort,features,splits}.parquet
Writes: results/metrics.json, results/results_table.md, results/slos_by_unit.csv,
        data/derived/test_predictions.parquet, data/derived/feature_importances.parquet
Usage : python src/evaluate.py   (or: make eval)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_config, load_dataset, make_splits, xy

# Upstream benchmark (Peres et al., 2022 / igor-peres repo README).
UPSTREAM = {"rmse": 4.02, "mae": 2.67, "r2_patient": 0.17, "r2_unit": 0.91}

# Small-cell suppression for published per-unit tables and figures: a unit
# with fewer stays than this is pooled into one "other units" row, because a
# per-unit total over one or two stays is effectively an individual's value.
MIN_CELL = 10


def suppress_small_units(unit, min_n=MIN_CELL):
    """Pool units with fewer than `min_n` admissions into a single row.
    Summary statistics (overall SLOS, unit-level R^2) are computed BEFORE
    suppression, on every unit, so published scalars are unchanged."""
    small = unit["admissions"] < min_n
    if not small.any():
        return unit
    pooled = unit[small][["admissions", "obs_total", "pred_total"]].sum()
    other = {"unitcode": f"Other units (each < {min_n} stays)",
             "admissions": int(pooled["admissions"]),
             "obs_total": pooled["obs_total"],
             "pred_total": pooled["pred_total"]}
    other["slos"] = other["obs_total"] / other["pred_total"]
    out = pd.concat([unit[~small], pd.DataFrame([other])], ignore_index=True)
    # the pooled row may itself be small; suppress its values if so
    if other["admissions"] < min_n:
        out = out[out["unitcode"] != other["unitcode"]]
    return out
CLIP = 21.0


def feature_importances(model):
    """Extract RF base-learner importances mapped to readable feature names."""
    rf_pipe = model.named_estimators_["rf"]
    prep = rf_pipe.named_steps["prep"]
    rf = rf_pipe.named_steps["rf"]
    names = list(prep.get_feature_names_out())
    # Strip the ColumnTransformer's "num__"/"cat__" prefixes for readability.
    names = [n.split("__", 1)[-1] for n in names]
    imp = pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def main():
    config = load_config()
    df = load_dataset(config)

    # Reuse the persisted split if present; otherwise recompute deterministically.
    splits_path = f"{config['derived_dir']}/splits.parquet"
    if os.path.exists(splits_path):
        sp = pd.read_parquet(splits_path)
        df = df.merge(sp, on="stay_id", how="left")
        test = df[df["split"] == "test"].copy()
    else:
        masks = make_splits(df, config)
        test = df[masks["test"]].copy()

    print(f"Evaluating on {len(test):,} held-out test stays "
          f"({test['subject_id'].nunique():,} patients).")

    bundle = joblib.load(f"{config['models_dir']}/stacking_model.joblib")
    model = bundle["model"]

    X_test, y_test = xy(test)
    y_pred = np.clip(model.predict(X_test), 0, CLIP)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Unit-level SLOS aggregation.
    pred_df = pd.DataFrame({
        "unitcode": test["unitcode"].values,
        "observed": y_test,
        "predicted": y_pred,
    })
    unit = (pred_df.groupby("unitcode")
            .agg(admissions=("observed", "size"),
                 obs_total=("observed", "sum"),
                 pred_total=("predicted", "sum"))
            .reset_index())
    unit["slos"] = unit["obs_total"] / unit["pred_total"]
    overall_slos = float(pred_df["observed"].sum() / pred_df["predicted"].sum())
    # Upstream unit-level R^2: predicted totals vs observed totals across units.
    r2_unit = float(r2_score(unit["obs_total"], unit["pred_total"]))

    metrics = {
        "cohort_n_total": int(len(df)),
        "test_n_stays": int(len(test)),
        "test_n_patients": int(test["subject_id"].nunique()),
        "patient_level": {"rmse": rmse, "mae": mae, "r2": r2},
        "unit_level": {"overall_slos": overall_slos, "r2": r2_unit,
                       "n_units": int(len(unit))},
        "upstream_benchmark": UPSTREAM,
        "base_rf_max_features": int(bundle.get("base_max_features", -1)),
    }

    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Aggregate-only SLOS table (per ICU unit, no patient rows; small units
    # pooled so that no published cell describes fewer than MIN_CELL stays).
    unit_sorted = suppress_small_units(unit).sort_values("admissions",
                                                         ascending=False)
    unit_sorted.to_csv(f"{results_dir}/slos_by_unit.csv", index=False)

    # Comparison table (committed; aggregate only).
    with open(f"{results_dir}/results_table.md", "w") as f:
        f.write("# Replication Results: MIMIC-IV v3.1 vs Upstream\n\n")
        f.write(f"Held-out test set: {len(test):,} ICU stays "
                f"({test['subject_id'].nunique():,} patients), "
                f"patient-level split (seed {config['split']['random_state']}).\n\n")
        f.write("| Metric | This replication (MIMIC-IV) | Upstream benchmark |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| RMSE (days) | {rmse:.3f} | {UPSTREAM['rmse']:.2f} |\n")
        f.write(f"| MAE (days) | {mae:.3f} | {UPSTREAM['mae']:.2f} |\n")
        f.write(f"| R^2 (patient-level) | {r2:.3f} | {UPSTREAM['r2_patient']:.2f} |\n")
        f.write(f"| R^2 (unit-level) | {r2_unit:.3f} | {UPSTREAM['r2_unit']:.2f} |\n")
        f.write(f"| Overall SLOS | {overall_slos:.3f} | 1.00 (by construction) |\n\n")
        f.write("## Per-ICU-unit SLOS\n\n")
        f.write("| ICU unit (first_careunit) | Admissions | Observed LOS (d) "
                "| Predicted LOS (d) | SLOS |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _, r in unit_sorted.iterrows():
            f.write(f"| {r['unitcode']} | {int(r['admissions']):,} | "
                    f"{r['obs_total']:.0f} | {r['pred_total']:.0f} | "
                    f"{r['slos']:.3f} |\n")

    # Persist per-patient predictions + importances for visualize.py (gitignored).
    pred_df.to_parquet(f"{config['derived_dir']}/test_predictions.parquet", index=False)
    imp = feature_importances(model)
    imp.to_parquet(f"{config['derived_dir']}/feature_importances.parquet", index=False)

    print("\n=== Patient-level ===")
    print(f"  RMSE = {rmse:.3f} d  (upstream {UPSTREAM['rmse']})")
    print(f"  MAE  = {mae:.3f} d  (upstream {UPSTREAM['mae']})")
    print(f"  R^2  = {r2:.3f}    (upstream {UPSTREAM['r2_patient']})")
    print("=== Unit-level (SLOS) ===")
    print(f"  overall SLOS = {overall_slos:.3f}")
    print(f"  R^2 (unit totals) = {r2_unit:.3f}  (upstream {UPSTREAM['r2_unit']})")
    print(f"\nTop features:\n{imp.head(10).to_string(index=False)}")
    print(f"\nSaved metrics -> {results_dir}/metrics.json, results_table.md, slos_by_unit.csv")


if __name__ == "__main__":
    main()
