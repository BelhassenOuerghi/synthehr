"""
multiseed.py
------------
Phase B of the evidence hardening: re-run Stages 0, 1 (release budget) and 2
(all arms) across N seeds on the CORRECTED patient-level splits, so that every
headline number acquires a confidence interval.

Why this exists. Stage 2 produced an accidental measurement of run-to-run
variation: `stage1` and `dpbaseline` are the *same recipe with fresh
randomness*, and they differed by 0.05 ROC-AUC / 0.10 R². Several single-seed
between-arm differences in the programme are smaller than that, i.e. they were
not yet findings. This driver replaces every headline figure with mean ± 95% CI.

Per seed it produces, for both tabular tasks:
    TRTR                 real-data ceiling (recomputed per seed: the split moves)
    stage0               non-private CTGAN                    (TSTR)
    stage1               DP-CTGAN at the release budget       (TSTR + MIA)
    dpbaseline           DP-CTGAN, fresh run (Stage-2 control)(TSTR)
    fairsample           DP + fair conditional sampling       (TSTR + fairness + MIA)
    fairtrain            DP + fair training-by-sampling       (TSTR + fairness)
plus bed demand (non-private and DP Poisson-AR).

Resumable: each seed writes `results/seeds/seed_<n>.json` on completion and is
skipped if that file exists, a crash at hour 15 costs one seed, not the run.
The primary seed additionally refreshes the canonical dataset folders, whose
current contents were generated from the leaking split and are stale.
"""
from __future__ import annotations

import argparse
import json
import os
import time
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
from ctgan import CTGAN
from dp_beddemand import dp_fit_poisson_ar
from fair_dp_ctgan import FairDPCTGAN
from fairness_metrics import (classification_fairness, regression_fairness,
                              representation)
from membership_inference import run_mia

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- downstream --
def _prep(disc, num):
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), disc)])


def _cols(df, tgt, discrete):
    feats = [c for c in df.columns if c != tgt]
    disc = [c for c in discrete if c != tgt and c in feats]
    return disc, [c for c in feats if c not in disc]


def score_los(train, test, tgt, disc, num, seed, sens):
    m = Pipeline([("p", _prep(disc, num)),
                  ("rf", RandomForestRegressor(n_estimators=200,
                                               min_samples_leaf=5, n_jobs=-1,
                                               random_state=seed))])
    m.fit(train[[c for c in train.columns if c != tgt]], train[tgt].clip(0, 21))
    pred = np.clip(m.predict(test[[c for c in test.columns if c != tgt]]), 0, 21)
    y = test[tgt].clip(0, 21)
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "fairness": {c: regression_fairness(y, pred, test[c], 200)
                         for c in sens}}


def score_readm(train, test, tgt, disc, num, seed, sens):
    from xgboost import XGBClassifier
    ytr = train[tgt].astype(int)
    pos, neg = max((ytr == 1).sum(), 1), max((ytr == 0).sum(), 1)
    m = Pipeline([("p", _prep(disc, num)),
                  ("x", XGBClassifier(n_estimators=300, max_depth=3,
                                      learning_rate=0.1, subsample=0.9,
                                      eval_metric="logloss",
                                      scale_pos_weight=neg / pos, n_jobs=-1,
                                      random_state=seed))])
    m.fit(train[[c for c in train.columns if c != tgt]], ytr)
    p = m.predict_proba(test[[c for c in test.columns if c != tgt]])[:, 1]
    y = test[tgt].astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
            "fairness": {c: classification_fairness(y, p, test[c], 200)
                         for c in sens}}


SCORERS = {"los": score_los, "readmission": score_readm}


# ---------------------------------------------------------------- generators --
def _fit_df(train, cfg):
    n = cfg["ctgan"]["max_train_rows"]
    return (train.sample(n, random_state=cfg["seed"]).reset_index(drop=True)
            if len(train) > n else train)


def gen_stage0(ds, cfg, train):
    c = cfg["ctgan"]
    print(f"    [stage0] fitting CTGAN ({c['epochs_stage0']} ep)...", flush=True)
    t = time.time()
    m = CTGAN(embedding_dim=c["embedding_dim"],
              generator_dim=tuple(c["generator_dim"]),
              discriminator_dim=tuple(c["discriminator_dim"]),
              batch_size=c["batch_size"], epochs=c["epochs_stage0"],
              pac=c["pac"], gp_lambda=c["gp_lambda"],
              vgm_components=c["vgm_components"], d_steps=c["d_steps"],
              seed=cfg["seed"], verbose=False).fit(_fit_df(train, cfg),
                                                   ds["discrete"])
    syn = m.sample(len(train))
    print(f"    [stage0] done in {time.time()-t:.0f}s", flush=True)
    return _restore(syn, train, ds["discrete"])


def gen_dp(ds, cfg, train, fair_cols, tag):
    c, dp = cfg["ctgan"], cfg["dp"]
    print(f"    [{tag}] fitting DP-CTGAN (eps={dp['primary_epsilon']}, "
          f"{c['epochs_dp']} ep)...", flush=True)
    t = time.time()
    m = FairDPCTGAN(fair_train_cols=fair_cols,
                    target_epsilon=dp["primary_epsilon"],
                    target_delta=dp["delta"], clip_norm=dp["clip_norm"],
                    hist_noise_multiplier=dp["hist_noise_multiplier"],
                    embedding_dim=c["embedding_dim"],
                    generator_dim=tuple(c["generator_dim"]),
                    discriminator_dim=tuple(c["discriminator_dim"]),
                    batch_size=c["batch_size"], epochs=c["epochs_dp"],
                    pac=c["pac"], gp_lambda=c["gp_lambda"],
                    vgm_components=c["vgm_components"], d_steps=c["d_steps"],
                    seed=cfg["seed"], verbose=False).fit(_fit_df(train, cfg),
                                                         ds["discrete"])
    print(f"    [{tag}] done in {time.time()-t:.0f}s "
          f"(eps spent {m.epsilon_spent_:.2f})", flush=True)
    return m


def _restore(syn, train, discrete):
    for c in train.columns:
        if c not in discrete:
            syn[c] = pd.to_numeric(syn[c], errors="coerce")
    return syn[train.columns]


# ---------------------------------------------------------------- one seed ----
def run_seed(cfg, seed):
    cfg = dict(cfg); cfg["seed"] = seed
    out = {"seed": seed, "tasks": {}}
    is_primary = (seed == cfg["primary_seed"])

    for loader, name in ((D.load_los, "los"),
                         (D.load_readmission, "readmission")):
        print(f"  == {name} (seed {seed}) ==", flush=True)
        ds = loader(cfg)
        tgt = ds["target"]
        sens = cfg["fairness"]["sensitive"][name]
        train = D.clean_numeric(ds["train"], ds["discrete"])
        test = D.clean_numeric(ds["test"], ds["discrete"])
        disc, num = _cols(train, tgt, ds["discrete"])
        score = SCORERS[name]
        arms, mia, eps = {}, {}, {}

        arms["TRTR"] = score(train, test, tgt, disc, num, seed, sens)

        s0 = gen_stage0(ds, cfg, train)
        arms["stage0"] = score(s0, test, tgt, disc, num, seed, sens)

        m1 = gen_dp(ds, cfg, train, [], "stage1")
        s1 = _restore(m1.sample(len(train)), train, ds["discrete"])
        arms["stage1"] = score(s1, test, tgt, disc, num, seed, sens)
        eps["stage1"] = m1.epsilon_spent_

        m2 = gen_dp(ds, cfg, train, [], "stage2-standard")
        plan = m2.build_plan(sens, cfg["fairness"]["min_support"])
        base = _restore(m2.sample(len(train)), train, ds["discrete"])
        fsam = _restore(m2.sample_fair(len(train), plan), train, ds["discrete"])
        arms["dpbaseline"] = score(base, test, tgt, disc, num, seed, sens)
        arms["fairsample"] = score(fsam, test, tgt, disc, num, seed, sens)
        eps["dpbaseline"] = eps["fairsample"] = m2.epsilon_spent_

        m3 = gen_dp(ds, cfg, train, sens, "stage2-fairtrain")
        plan3 = m3.build_plan(sens, cfg["fairness"]["min_support"])
        ftr = _restore(m3.sample_fair(len(train), plan3), train, ds["discrete"])
        arms["fairtrain"] = score(ftr, test, tgt, disc, num, seed, sens)
        eps["fairtrain"] = m3.epsilon_spent_

        # membership inference, now on PATIENT-DISJOINT members/non-members
        mia_disc = [c for c in ds["discrete"] if c != tgt]
        nc = cfg["mia"]["n_candidates"]
        for arm, syn in (("stage0", s0), ("stage1", s1), ("fairsample", fsam)):
            mia[arm] = run_mia(train, test, syn, mia_disc,
                               n_candidates=nc, seed=seed)
        print(f"    [mia] done", flush=True)

        rep = {a: representation(df, train, sens) for a, df in
               (("stage0", s0), ("dpbaseline", base), ("fairsample", fsam),
                ("fairtrain", ftr))}
        out["tasks"][name] = {"arms": arms, "mia": mia, "epsilon": eps,
                              "representation": rep,
                              "n_train": len(train), "n_test": len(test)}

        if is_primary:      # refresh the stale canonical datasets
            for root, df in ((cfg["stage0_root"], s0), (cfg["stage1_root"], s1),
                             (cfg["stage2_root"], fsam)):
                os.makedirs(root, exist_ok=True)
                df.to_parquet(os.path.join(root, f"{name}_synthetic.parquet"),
                              index=False)
            print(f"    [primary] refreshed canonical {name} datasets",
                  flush=True)

    # ---- bed demand (cheap; no sensitive attribute) ------------------------
    bd = D.load_beddemand(cfg)
    W = cfg["beddemand"]["ar_window"]
    rng = np.random.default_rng(seed)
    h = len(bd["y_test"])

    def fc_score(series):
        beta = BD.fit_poisson_ar(series, W)
        f = BD.forecast(beta, bd["y_train"], h, W)
        return {"mae": float(np.mean(np.abs(bd["y_test"] - f))),
                "rmse": float(np.sqrt(np.mean((bd["y_test"] - f) ** 2)))}

    b0 = BD.fit_poisson_ar(bd["y_train"], W)
    s0_bd = BD.sample_series(b0, bd["y_train"][:W], len(bd["y_train"]), W, rng)
    bdp, eps_bd, _ = dp_fit_poisson_ar(
        bd["y_train"], W, target_epsilon=cfg["dp"]["primary_epsilon"],
        target_delta=cfg["dp"]["delta_beddemand"],
        clip_norm=cfg["dp"]["clip_norm"], iters=cfg["dp"]["beddemand_iters"],
        seed=seed, verbose=False)
    s1_bd = BD.sample_series(bdp, bd["y_train"][:W], len(bd["y_train"]), W, rng)
    out["beddemand"] = {"TRTR": fc_score(bd["y_train"]),
                        "stage0": fc_score(s0_bd), "stage1": fc_score(s1_bd),
                        "epsilon_stage1": eps_bd}
    if is_primary:
        pd.DataFrame({"count": s1_bd.astype(int)}).to_csv(
            os.path.join(cfg["stage1_root"], "beddemand_synthetic.csv"),
            index=False)
        pd.DataFrame({"count": s0_bd.astype(int)}).to_csv(
            os.path.join(cfg["stage0_root"], "beddemand_synthetic.csv"),
            index=False)
    return out


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sdir = os.path.join(HERE, "results", "seeds")
    os.makedirs(sdir, exist_ok=True)
    for seed in cfg["seeds"]:
        path = os.path.join(sdir, f"seed_{seed}.json")
        if os.path.exists(path):
            print(f"seed {seed}: already done, skipping", flush=True)
            continue
        print(f"\n=== SEED {seed} ===", flush=True)
        t = time.time()
        res = run_seed(cfg, seed)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"=== SEED {seed} complete in {(time.time()-t)/60:.0f} min "
              f"-> {path} ===", flush=True)
    print("\nall seeds complete.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    main(ap.parse_args().config)
