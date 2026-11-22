"""
evaluate_utility.py
-------------------
The core comparison: train-on-real / test-on-real (TRTR) vs
train-on-synthetic / test-on-real (TSTR).

Protocol (identical downstream model in both arms; only the *training* set
differs, real vs synthetic):
  * LOS         RandomForestRegressor  -> RMSE / MAE / R²
  * readmission XGBoost classifier     -> ROC-AUC / PR-AUC / accuracy
  * bed demand  Poisson-AR forecaster  -> MAE / RMSE on the real test span
The real test set is NEVER seen during synthesis, so a small TRTR-TSTR gap is
direct evidence that the synthetic data preserved the predictive signal.
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

import data as D
import beddemand_gen as BD

warnings.filterwarnings("ignore")


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


def eval_los(cfg, seed):
    ds = D.load_los(cfg)
    real_tr = D.clean_numeric(ds["train"], ds["discrete"])
    test = D.clean_numeric(ds["test"], ds["discrete"])
    syn = pd.read_parquet(os.path.join(cfg["synthetic_root"], "los_synthetic.parquet"))
    tgt = ds["target"]
    Xr, disc, num = _xy(real_tr, tgt, ds["discrete"])
    res = {}
    for arm, train in (("TRTR", real_tr), ("TSTR", syn)):
        Xtr, _, _ = _xy(train, tgt, ds["discrete"])
        model = Pipeline([("prep", _prep(disc, num)),
                          ("rf", RandomForestRegressor(
                              n_estimators=200, min_samples_leaf=5,
                              n_jobs=-1, random_state=seed))])
        model.fit(Xtr, train[tgt].clip(0, 21))
        pred = np.clip(model.predict(test[[c for c in test.columns if c != tgt]]), 0, 21)
        yte = test[tgt].clip(0, 21)
        res[arm] = {
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred)),
        }
    res["delta_r2_tstr_minus_trtr"] = res["TSTR"]["r2"] - res["TRTR"]["r2"]
    return res


def eval_readmission(cfg, seed):
    from xgboost import XGBClassifier
    ds = D.load_readmission(cfg)
    real_tr = D.clean_numeric(ds["train"], ds["discrete"])
    test = D.clean_numeric(ds["test"], ds["discrete"])
    syn = pd.read_parquet(os.path.join(cfg["synthetic_root"],
                                       "readmission_synthetic.parquet"))
    tgt = ds["target"]
    _, disc, num = _xy(real_tr, tgt, ds["discrete"])
    res = {}
    for arm, train in (("TRTR", real_tr), ("TSTR", syn)):
        ytr = train[tgt].astype(int)
        Xtr = train[[c for c in train.columns if c != tgt]]
        pos = max((ytr == 1).sum(), 1); neg = max((ytr == 0).sum(), 1)
        model = Pipeline([("prep", _prep(disc, num)),
                          ("xgb", XGBClassifier(
                              n_estimators=300, max_depth=3, learning_rate=0.1,
                              subsample=0.9, eval_metric="logloss",
                              scale_pos_weight=neg / pos, n_jobs=-1,
                              random_state=seed))])
        model.fit(Xtr, ytr)
        Xte = test[[c for c in test.columns if c != tgt]]
        proba = model.predict_proba(Xte)[:, 1]
        yte = test[tgt].astype(int)
        res[arm] = {
            "roc_auc": float(roc_auc_score(yte, proba)),
            "pr_auc": float(average_precision_score(yte, proba)),
            "accuracy": float(accuracy_score(yte, (proba > 0.5).astype(int))),
        }
    res["delta_auc_tstr_minus_trtr"] = res["TSTR"]["roc_auc"] - res["TRTR"]["roc_auc"]
    return res


def eval_beddemand(cfg, seed):
    ds = D.load_beddemand(cfg)
    W = cfg["beddemand"]["ar_window"]
    syn = pd.read_csv(os.path.join(cfg["synthetic_root"],
                                   "beddemand_synthetic.csv"))["count"].to_numpy(float)
    horizon = len(ds["y_test"])
    res = {}
    for arm, train_series in (("TRTR", ds["y_train"]), ("TSTR", syn)):
        beta = BD.fit_poisson_ar(train_series, W)
        # forecast the real test span from the real train boundary
        fc = BD.forecast(beta, ds["y_train"], horizon, W)
        res[arm] = {
            "mae": float(np.mean(np.abs(ds["y_test"] - fc))),
            "rmse": float(np.sqrt(np.mean((ds["y_test"] - fc) ** 2))),
        }
    res["delta_mae_tstr_minus_trtr"] = res["TSTR"]["mae"] - res["TRTR"]["mae"]
    res["real_train_mean"] = float(np.mean(ds["y_train"]))
    res["synthetic_mean"] = float(np.mean(syn))
    return res


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    print("evaluating LOS...", flush=True);          los = eval_los(cfg, seed)
    print("evaluating readmission...", flush=True);  rd = eval_readmission(cfg, seed)
    print("evaluating bed demand...", flush=True);   bd = eval_beddemand(cfg, seed)
    metrics = {"protocol": "TRTR vs TSTR; identical model per task, real test set",
               "los": los, "readmission": rd, "beddemand": bd}
    os.makedirs(cfg["results_dir"] if "results_dir" in cfg else "results",
                exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _write_table(metrics)
    print("wrote results/metrics.json and results/results_table.md", flush=True)


def _write_table(m):
    L = ["# Utility results: TRTR vs TSTR (real test set)", ""]
    los = m["los"]
    L += ["## Length of Stay (regression): RandomForest", "",
          "| Arm | RMSE (d) | MAE (d) | R² |", "|---|---|---|---|",
          f"| TRTR (train real) | {los['TRTR']['rmse']:.3f} | {los['TRTR']['mae']:.3f} | {los['TRTR']['r2']:.3f} |",
          f"| TSTR (train synthetic) | {los['TSTR']['rmse']:.3f} | {los['TSTR']['mae']:.3f} | {los['TSTR']['r2']:.3f} |",
          f"| Δ R² (TSTR − TRTR) | | | {los['delta_r2_tstr_minus_trtr']:+.3f} |", ""]
    rd = m["readmission"]
    L += ["## 30-day Readmission (classification): XGBoost", "",
          "| Arm | ROC-AUC | PR-AUC | Accuracy |", "|---|---|---|---|",
          f"| TRTR (train real) | {rd['TRTR']['roc_auc']:.3f} | {rd['TRTR']['pr_auc']:.3f} | {rd['TRTR']['accuracy']:.3f} |",
          f"| TSTR (train synthetic) | {rd['TSTR']['roc_auc']:.3f} | {rd['TSTR']['pr_auc']:.3f} | {rd['TSTR']['accuracy']:.3f} |",
          f"| Δ ROC-AUC (TSTR − TRTR) | {rd['delta_auc_tstr_minus_trtr']:+.3f} | | |", ""]
    bd = m["beddemand"]
    L += ["## Bed demand (forecast): Poisson-AR", "",
          "| Arm | MAE | RMSE |", "|---|---|---|",
          f"| TRTR (train real) | {bd['TRTR']['mae']:.3f} | {bd['TRTR']['rmse']:.3f} |",
          f"| TSTR (train synthetic) | {bd['TSTR']['mae']:.3f} | {bd['TSTR']['rmse']:.3f} |",
          f"| Δ MAE (TSTR − TRTR) | {bd['delta_mae_tstr_minus_trtr']:+.3f} | |", ""]
    L[1:1] = ["", '> Single training run (seed 42) on patient-disjoint splits, kept for provenance. The authoritative numbers are the five-seed means and 95% intervals in [research/evidence](../../evidence): a single training run is not representative of its recipe.']
    with open("results/results_table.md", "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
