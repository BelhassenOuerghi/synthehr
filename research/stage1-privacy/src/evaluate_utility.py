"""
evaluate_utility.py
-------------------
Stage-1 evaluation = the Stage-0 protocol plus the privacy dimension:

  * utility, TRTR vs TSTR(ε) for every ε in the grid, with the identical
               downstream model per task (LOS RandomForest, readmission
               XGBoost, bed-demand Poisson-AR forecaster), scored on the SAME
               untouched real test set as Stage 0;
  * privacy, membership-inference attacks (DCR + k-NN classifier) against
               each ε's synthetic release, plus copy-detection statistics;
  * context, the Stage-0 (non-private CTGAN) TSTR numbers are read from the
               baseline repo's results/metrics.json, so every table shows
               TRTR ceiling / Stage-0 TSTR / Stage-1 TSTR(ε) side by side.
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
from membership_inference import run_mia

warnings.filterwarnings("ignore")


def _eps_tag(eps):
    return f"eps{eps:g}".replace(".", "p")


def _prep(discrete, numeric):
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, discrete)])


def _xy(df, target, discrete):
    feats = [c for c in df.columns if c != target]
    disc = [c for c in discrete if c != target and c in feats]
    num = [c for c in feats if c not in disc]
    return df[feats], disc, num


def _los_metrics(train, test, tgt, disc, num, seed):
    model = Pipeline([("prep", _prep(disc, num)),
                      ("rf", RandomForestRegressor(
                          n_estimators=200, min_samples_leaf=5,
                          n_jobs=-1, random_state=seed))])
    model.fit(train[[c for c in train.columns if c != tgt]],
              train[tgt].clip(0, 21))
    pred = np.clip(model.predict(
        test[[c for c in test.columns if c != tgt]]), 0, 21)
    yte = test[tgt].clip(0, 21)
    return {"rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred))}


def _readm_metrics(train, test, tgt, disc, num, seed):
    from xgboost import XGBClassifier
    ytr = train[tgt].astype(int)
    pos = max((ytr == 1).sum(), 1); neg = max((ytr == 0).sum(), 1)
    model = Pipeline([("prep", _prep(disc, num)),
                      ("xgb", XGBClassifier(
                          n_estimators=300, max_depth=3, learning_rate=0.1,
                          subsample=0.9, eval_metric="logloss",
                          scale_pos_weight=neg / pos, n_jobs=-1,
                          random_state=seed))])
    model.fit(train[[c for c in train.columns if c != tgt]], ytr)
    proba = model.predict_proba(
        test[[c for c in test.columns if c != tgt]])[:, 1]
    yte = test[tgt].astype(int)
    return {"roc_auc": float(roc_auc_score(yte, proba)),
            "pr_auc": float(average_precision_score(yte, proba)),
            "accuracy": float(accuracy_score(yte, (proba > 0.5).astype(int)))}


def eval_tabular(cfg, loader, metric_fn, seed):
    ds = loader(cfg)
    tgt = ds["target"]
    real_tr = D.clean_numeric(ds["train"], ds["discrete"])
    test = D.clean_numeric(ds["test"], ds["discrete"])
    _, disc, num = _xy(real_tr, tgt, ds["discrete"])
    out = {"TRTR": metric_fn(real_tr, test, tgt, disc, num, seed),
           "TSTR_by_epsilon": {}, "MIA_by_epsilon": {}}
    for eps in cfg["dp"]["epsilon_grid"]:
        path = os.path.join(cfg["synthetic_root"],
                            f"{ds['name']}_synthetic_{_eps_tag(eps)}.parquet")
        syn = pd.read_parquet(path)
        out["TSTR_by_epsilon"][str(eps)] = metric_fn(syn, test, tgt, disc,
                                                     num, seed)
        out["MIA_by_epsilon"][str(eps)] = run_mia(
            real_tr, test, syn, [c for c in ds["discrete"] if c != tgt],
            n_candidates=cfg["mia"]["n_candidates"], seed=seed)
        print(f"  [{ds['name']}] eps={eps} done", flush=True)
    return out


def eval_beddemand(cfg, seed):
    ds = D.load_beddemand(cfg)
    W = cfg["beddemand"]["ar_window"]
    horizon = len(ds["y_test"])

    def score(train_series):
        beta = BD.fit_poisson_ar(train_series, W)
        fc = BD.forecast(beta, ds["y_train"], horizon, W)
        return {"mae": float(np.mean(np.abs(ds["y_test"] - fc))),
                "rmse": float(np.sqrt(np.mean((ds["y_test"] - fc) ** 2)))}

    out = {"TRTR": score(ds["y_train"]), "TSTR_by_epsilon": {}}
    for eps in cfg["dp"]["epsilon_grid"]:
        syn = pd.read_csv(os.path.join(
            cfg["synthetic_root"],
            f"beddemand_synthetic_{_eps_tag(eps)}.csv"))["count"].to_numpy(float)
        out["TSTR_by_epsilon"][str(eps)] = score(syn)
    return out


def _stage0_reference(cfg):
    p = cfg.get("stage0_metrics")
    if p and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    print("evaluating LOS...", flush=True)
    los = eval_tabular(cfg, D.load_los, _los_metrics, seed)
    print("evaluating readmission...", flush=True)
    rd = eval_tabular(cfg, D.load_readmission, _readm_metrics, seed)
    print("evaluating bed demand...", flush=True)
    bd = eval_beddemand(cfg, seed)
    metrics = {
        "protocol": "TRTR vs TSTR(eps) grid; identical model per task, real "
                    "test set; MIA (DCR + kNN classifier) per eps",
        "stage0_reference": _stage0_reference(cfg),
        "los": los, "readmission": rd, "beddemand": bd,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_table(metrics, cfg)
    print("wrote results/metrics.json and results/results_table.md", flush=True)


def _write_table(m, cfg):
    grid = [str(e) for e in cfg["dp"]["epsilon_grid"]]
    s0 = m.get("stage0_reference") or {}
    L = ["# Stage 1: privacy to utility results (TRTR vs TSTR(ε), real test set)",
         ""]
    los, s0los = m["los"], s0.get("los", {})
    L += ["## Length of Stay (regression): RandomForest", "",
          "| Arm | RMSE (d) | MAE (d) | R² | MIA AUC (DCR / kNN) |",
          "|---|---|---|---|---|",
          f"| TRTR (ceiling) | {los['TRTR']['rmse']:.3f} | "
          f"{los['TRTR']['mae']:.3f} | {los['TRTR']['r2']:.3f} | n/a |"]
    if s0los:
        L += [f"| Stage-0 TSTR (ε = ∞) | {s0los['TSTR']['rmse']:.3f} | "
              f"{s0los['TSTR']['mae']:.3f} | {s0los['TSTR']['r2']:.3f} | n/a |"]
    for e in grid:
        t, a = los["TSTR_by_epsilon"][e], los["MIA_by_epsilon"][e]
        L += [f"| TSTR (ε = {e}) | {t['rmse']:.3f} | {t['mae']:.3f} | "
              f"{t['r2']:.3f} | {a['attack_auc_dcr']:.3f} / "
              f"{a['attack_auc_knn_classifier']:.3f} |"]
    L += [""]
    rd, s0rd = m["readmission"], s0.get("readmission", {})
    L += ["## 30-day Readmission (classification): XGBoost", "",
          "| Arm | ROC-AUC | PR-AUC | Accuracy | MIA AUC (DCR / kNN) |",
          "|---|---|---|---|---|",
          f"| TRTR (ceiling) | {rd['TRTR']['roc_auc']:.3f} | "
          f"{rd['TRTR']['pr_auc']:.3f} | {rd['TRTR']['accuracy']:.3f} | n/a |"]
    if s0rd:
        L += [f"| Stage-0 TSTR (ε = ∞) | {s0rd['TSTR']['roc_auc']:.3f} | "
              f"{s0rd['TSTR']['pr_auc']:.3f} | "
              f"{s0rd['TSTR']['accuracy']:.3f} | n/a |"]
    for e in grid:
        t, a = rd["TSTR_by_epsilon"][e], rd["MIA_by_epsilon"][e]
        L += [f"| TSTR (ε = {e}) | {t['roc_auc']:.3f} | {t['pr_auc']:.3f} | "
              f"{t['accuracy']:.3f} | {a['attack_auc_dcr']:.3f} / "
              f"{a['attack_auc_knn_classifier']:.3f} |"]
    L += [""]
    bd, s0bd = m["beddemand"], s0.get("beddemand", {})
    L += ["## Bed demand (forecast): Poisson-AR", "",
          "| Arm | MAE | RMSE |", "|---|---|---|",
          f"| TRTR (ceiling) | {bd['TRTR']['mae']:.3f} | "
          f"{bd['TRTR']['rmse']:.3f} |"]
    if s0bd:
        L += [f"| Stage-0 TSTR (ε = ∞) | {s0bd['TSTR']['mae']:.3f} | "
              f"{s0bd['TSTR']['rmse']:.3f} |"]
    for e in grid:
        t = bd["TSTR_by_epsilon"][e]
        L += [f"| TSTR (ε = {e}) | {t['mae']:.3f} | {t['rmse']:.3f} |"]
    L[1:1] = ["", '> Single training run (seed 42) on patient-disjoint splits, kept for provenance. The authoritative numbers are the five-seed means and 95% intervals in [research/evidence](../../evidence): a single training run is not representative of its recipe.']
    with open("results/results_table.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
