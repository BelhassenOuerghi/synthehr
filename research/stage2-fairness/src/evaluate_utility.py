"""
evaluate_utility.py
-------------------
Stage-2 evaluation = the frozen TSTR/TRTR harness + the fairness dimension:

  arms per tabular task (all scored on the SAME real test set):
    TRTR, the ceiling, and the disparity a real-data model has;
    stage0 (ε = ∞), non-private CTGAN release (reference);
    stage1 (ε = 10), DP, no fairness intervention (reference);
    dpbaseline, this stage's DP control (fresh training);
    fairsample, DP + fair conditional sampling (zero extra ε);
    fairtrain, DP + fair training-by-sampling + fair sampling.

  per arm: utility (task metrics) + subgroup fairness (per sensitive column:
  auc_gap / dp_diff / eo_diff for classification, rmse_gap / bias_gap for
  regression) + representation shares; MIA on the canonical arm.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import beddemand_gen as BD
import data as D
from fairness_metrics import (classification_fairness, regression_fairness,
                              representation)
from membership_inference import run_mia

warnings.filterwarnings("ignore")


def _prep(discrete, numeric):
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, discrete)])


def _split_cols(df, target, discrete):
    feats = [c for c in df.columns if c != target]
    disc = [c for c in discrete if c != target and c in feats]
    num = [c for c in feats if c not in disc]
    return disc, num


def _arm_paths(cfg, name):
    root = cfg["synthetic_root"]
    return {
        "stage0": os.path.join(cfg["stage0_root"], f"{name}_synthetic.parquet"),
        "stage1": os.path.join(cfg["stage1_root"], f"{name}_synthetic.parquet"),
        "dpbaseline": os.path.join(root, f"{name}_synthetic_dpbaseline.parquet"),
        "fairsample": os.path.join(root, f"{name}_synthetic_fairsample.parquet"),
        "fairtrain": os.path.join(root, f"{name}_synthetic_fairtrain.parquet"),
    }


def eval_los(cfg, seed):
    ds = D.load_los(cfg)
    tgt, sens = ds["target"], cfg["fairness"]["sensitive"]["los"]
    real_tr = D.clean_numeric(ds["train"], ds["discrete"])
    test = D.clean_numeric(ds["test"], ds["discrete"])
    disc, num = _split_cols(real_tr, tgt, ds["discrete"])
    Xte = test[[c for c in test.columns if c != tgt]]
    yte = test[tgt].clip(0, 21)
    out = {}
    arms = {"TRTR": real_tr, **{a: pd.read_parquet(p)
                                for a, p in _arm_paths(cfg, "los").items()}}
    for arm, train in arms.items():
        model = Pipeline([("prep", _prep(disc, num)),
                          ("rf", RandomForestRegressor(
                              n_estimators=200, min_samples_leaf=5,
                              n_jobs=-1, random_state=seed))])
        model.fit(train[[c for c in train.columns if c != tgt]],
                  train[tgt].clip(0, 21))
        pred = np.clip(model.predict(Xte), 0, 21)
        out[arm] = {
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred)),
            "fairness": {c: regression_fairness(
                yte, pred, test[c], cfg["fairness"]["min_group_n"])
                for c in sens},
            "representation": representation(train, real_tr, sens),
        }
        print(f"  [los] {arm} done", flush=True)
    return out


def eval_readmission(cfg, seed):
    from xgboost import XGBClassifier
    ds = D.load_readmission(cfg)
    tgt, sens = ds["target"], cfg["fairness"]["sensitive"]["readmission"]
    real_tr = D.clean_numeric(ds["train"], ds["discrete"])
    test = D.clean_numeric(ds["test"], ds["discrete"])
    disc, num = _split_cols(real_tr, tgt, ds["discrete"])
    Xte = test[[c for c in test.columns if c != tgt]]
    yte = test[tgt].astype(int)
    out = {}
    arms = {"TRTR": real_tr, **{a: pd.read_parquet(p) for a, p in
                                _arm_paths(cfg, "readmission").items()}}
    for arm, train in arms.items():
        ytr = train[tgt].astype(int)
        pos = max((ytr == 1).sum(), 1); neg = max((ytr == 0).sum(), 1)
        model = Pipeline([("prep", _prep(disc, num)),
                          ("xgb", XGBClassifier(
                              n_estimators=300, max_depth=3, learning_rate=0.1,
                              subsample=0.9, eval_metric="logloss",
                              scale_pos_weight=neg / pos, n_jobs=-1,
                              random_state=seed))])
        model.fit(train[[c for c in train.columns if c != tgt]], ytr)
        proba = model.predict_proba(Xte)[:, 1]
        out[arm] = {
            "roc_auc": float(roc_auc_score(yte, proba)),
            "pr_auc": float(average_precision_score(yte, proba)),
            "accuracy": float(accuracy_score(yte, (proba > 0.5).astype(int))),
            "fairness": {c: classification_fairness(
                yte, proba, test[c], cfg["fairness"]["min_group_n"])
                for c in sens},
            "representation": representation(train, real_tr, sens),
        }
        print(f"  [readmission] {arm} done", flush=True)
    return out


def eval_beddemand(cfg):
    ds = D.load_beddemand(cfg)
    W = cfg["beddemand"]["ar_window"]
    horizon = len(ds["y_test"])

    def score(series):
        beta = BD.fit_poisson_ar(series, W)
        fc = BD.forecast(beta, ds["y_train"], horizon, W)
        return {"mae": float(np.mean(np.abs(ds["y_test"] - fc))),
                "rmse": float(np.sqrt(np.mean((ds["y_test"] - fc) ** 2)))}

    syn = pd.read_csv(os.path.join(cfg["synthetic_root"],
                                   "beddemand_synthetic.csv"))["count"]
    return {"TRTR": score(ds["y_train"]), "TSTR": score(syn.to_numpy(float))}


def eval_mia(cfg, seed):
    out = {}
    for loader, name in ((D.load_los, "los"),
                         (D.load_readmission, "readmission")):
        ds = loader(cfg)
        tr = D.clean_numeric(ds["train"], ds["discrete"])
        te = D.clean_numeric(ds["test"], ds["discrete"])
        syn = pd.read_parquet(os.path.join(cfg["synthetic_root"],
                                           f"{name}_synthetic.parquet"))
        out[name] = run_mia(tr, te, syn,
                            [c for c in ds["discrete"] if c != ds["target"]],
                            n_candidates=cfg["mia"]["n_candidates"], seed=seed)
        print(f"  [mia] {name} done", flush=True)
    return out


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    print("evaluating LOS...", flush=True)
    los = eval_los(cfg, seed)
    print("evaluating readmission...", flush=True)
    rd = eval_readmission(cfg, seed)
    print("evaluating bed demand...", flush=True)
    bd = eval_beddemand(cfg)
    print("membership inference on canonical arm...", flush=True)
    mia = eval_mia(cfg, seed)
    metrics = {"protocol": "TRTR + 5 synthetic arms; identical model per task; "
                           "real test set; subgroup fairness + representation; "
                           "MIA on canonical arm",
               "canonical_arm": cfg["canonical_arm"],
               "los": los, "readmission": rd, "beddemand": bd, "mia": mia}
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_table(metrics, cfg)
    print("wrote results/metrics.json and results/results_table.md", flush=True)


def _fmt(x, nd=3):
    return "," if x is None else f"{x:.{nd}f}"


def _write_table(m, cfg):
    ARMS = ["TRTR", "stage0", "stage1", "dpbaseline", "fairsample", "fairtrain"]
    L = ["# Stage 2: fairness to utility results (real test set)", ""]
    L += ["## 30-day readmission: XGBoost (sensitive: gender, insurance)", "",
          "| Arm | ROC-AUC | PR-AUC | AUC gap (gen / ins) | DP diff (gen / ins) | EO diff (gen / ins) |",
          "|---|---|---|---|---|---|"]
    for a in ARMS:
        r = m["readmission"][a]
        fg, fi = r["fairness"].get("gender", {}), r["fairness"].get("insurance", {})
        L += [f"| {a} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} "
              f"| {_fmt(fg.get('auc_gap'))} / {_fmt(fi.get('auc_gap'))} "
              f"| {_fmt(fg.get('dp_diff'))} / {_fmt(fi.get('dp_diff'))} "
              f"| {_fmt(fg.get('eo_diff'))} / {_fmt(fi.get('eo_diff'))} |"]
    L += ["", "## Length of stay: RandomForest (sensitive: gender)", "",
          "| Arm | RMSE (d) | R² | RMSE gap | bias gap |", "|---|---|---|---|---|"]
    for a in ARMS:
        r = m["los"][a]
        f = r["fairness"].get("gender", {})
        L += [f"| {a} | {r['rmse']:.3f} | {r['r2']:.3f} "
              f"| {_fmt(f.get('rmse_gap'))} | {_fmt(f.get('bias_gap'))} |"]
    bd = m["beddemand"]
    L += ["", "## Bed demand: Poisson-AR (DP pass-through)", "",
          "| Arm | MAE | RMSE |", "|---|---|---|",
          f"| TRTR | {bd['TRTR']['mae']:.3f} | {bd['TRTR']['rmse']:.3f} |",
          f"| TSTR (stage 2) | {bd['TSTR']['mae']:.3f} | {bd['TSTR']['rmse']:.3f} |"]
    L += ["", "## Membership inference: canonical arm "
          f"(`{m['canonical_arm']}`)", "",
          "| Task | DCR AUC | kNN AUC | near-dup@0.1 |", "|---|---|---|---|"]
    for t, r in m["mia"].items():
        L += [f"| {t} | {r['attack_auc_dcr']:.3f} "
              f"| {r['attack_auc_knn_classifier']:.3f} "
              f"| {r['near_duplicate_share@0.1']:.3f} |"]
    L[1:1] = ["", '> Single training run (seed 42) on patient-disjoint splits, kept for provenance. The authoritative numbers are the five-seed means and 95% intervals in [research/evidence](../../evidence): a single training run is not representative of its recipe.']
    with open("results/results_table.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
