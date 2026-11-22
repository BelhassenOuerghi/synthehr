"""
jobs.py
-------
One unit of the Pillar 4 experiment: a (task, seed, arm) triple. A generator
arm fits a generator on the frozen harness's 30,000-row fitting sample, writes
a synthetic dataset the size of the real training set to the synthetic-data
folder (never this repository), and scores it. A pooled arm assembles a
release from component datasets that already exist. Every arm is scored the
same way:

  utility         frozen downstream model, trained on the arm's data and
                  tested on the untouched real test split (TSTR), with the
                  frozen subgroup-fairness metrics;
  mia             the frozen membership-inference attacks (patient-disjoint
                  members and non-members), plus the low-false-positive tail;
  fidelity        realism against real rows the generator never saw;
  representation  subgroup shares versus the real training data;
  privacy         every run's accounting parameters, so pooled releases can
                  be composed exactly.

Model seeds. Replicate r of a recipe uses model seed (data seed + 1000 r) on
the SAME fitting rows, so replicates differ only in model randomness. The
first DP-CTGAN run (r = 0) is the evidence repository's `stage1` recipe and
seed, and reproduces it.

Usage: python jobs.py --task los --seed 42 --arm dpddpm --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

import budget
import realism_data as RD
import settings
from ctgan import CTGAN
from fair_dp_ctgan import FairDPCTGAN
from fair_resample import fair_resample
from fairness_metrics import representation
from fidelity import fidelity, mia_tail
from membership_inference import run_mia
from multiseed import SCORERS, _cols, _restore
from privacy_accountant import compose_epsilon
from tabddpm import DPTabDDPM, TabDDPM

warnings.filterwarnings("ignore")
HERE = settings.HERE

K = 4
GEN = {"ctgan": dict(family="ctgan", dp=False),
       "dpctgan": dict(family="ctgan", dp=True, rep=0, rake=True),
       "ddpm": dict(family="ddpm", dp=False, cap="M"),
       "ddpm_S": dict(family="ddpm", dp=False, cap="S"),
       "ddpm_L": dict(family="ddpm", dp=False, cap="L"),
       "dpddpm": dict(family="ddpm", dp=True, cap="M", rep=0, rake=True),
       "dpddpm_patient": dict(family="ddpm", dp=True, cap="M", rep=0,
                              unit="patient"),
       "dpddpm_S": dict(family="ddpm", dp=True, cap="S", rep=0),
       "dpddpm_L": dict(family="ddpm", dp=True, cap="L", rep=0)}
for _r in range(1, K):
    GEN[f"dpctgan_r{_r}"] = dict(family="ctgan", dp=True, rep=_r)
    GEN[f"dpddpm_r{_r}"] = dict(family="ddpm", dp=True, cap="M", rep=_r)
for _r in range(1, K + 1):
    GEN[f"dpctgan_s{_r}"] = dict(family="ctgan", dp=True, rep=10 + _r,
                                 split=K)
    GEN[f"dpddpm_s{_r}"] = dict(family="ddpm", dp=True, cap="M",
                                rep=10 + _r, split=K)
POOLED = {
    "dpctgan_ens4": ["dpctgan"] + [f"dpctgan_r{r}" for r in range(1, K)],
    "dpctgan_ens4split": [f"dpctgan_s{r}" for r in range(1, K + 1)],
    "dpddpm_ens4": ["dpddpm"] + [f"dpddpm_r{r}" for r in range(1, K)],
    "dpddpm_ens4split": [f"dpddpm_s{r}" for r in range(1, K + 1)],
}
DERIVED = {"dpctgan_rake": "dpctgan", "dpddpm_rake": "dpddpm"}
ALL_ARMS = ["TRTR"] + list(GEN) + list(POOLED) + list(DERIVED)


def device_of(arm):
    if arm in GEN and GEN[arm]["family"] == "ddpm":
        return "mps"
    return "cpu"


def deps_of(arm):
    if arm in DERIVED:
        return [DERIVED[arm]]
    return POOLED.get(arm, [])


# ------------------------------------------------------------------ paths --
def result_path(task, seed, arm):
    root = os.environ.get("SYNTHEHR_RESULTS_DIR", os.path.join(HERE, "results"))
    return os.path.join(root, "arms", task, f"seed_{seed}", f"{arm}.json")


def data_path(cfg, task, seed, arm):
    return os.path.join(cfg["synthetic_root"], "work", task, f"seed_{seed}",
                        f"{arm}.parquet")


def _save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _write(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


# ------------------------------------------------------------- generators --
def _ctgan_kwargs(cfg):
    c = cfg["ctgan"]
    return dict(embedding_dim=c["embedding_dim"],
                generator_dim=tuple(c["generator_dim"]),
                discriminator_dim=tuple(c["discriminator_dim"]),
                batch_size=c["batch_size"], pac=c["pac"],
                gp_lambda=c["gp_lambda"], vgm_components=c["vgm_components"],
                d_steps=c["d_steps"], verbose=False)


def fit_ctgan(cfg, d, spec, seed):
    kw = _ctgan_kwargs(cfg)
    if not spec["dp"]:
        m = CTGAN(epochs=cfg["ctgan"]["epochs_stage0"], seed=seed,
                  **kw).fit(d["fit"], d["discrete"])
        return m, []
    dp, n = cfg["dp"], len(d["fit"])
    batch = (kw["batch_size"] // kw["pac"]) * kw["pac"]
    q = batch / n
    steps = cfg["ctgan"]["epochs_dp"] * max(n // batch, 1) * kw["d_steps"]
    sig = None
    if spec.get("split"):
        sig = budget.calibrate_split(dp["primary_epsilon"], q, steps,
                                     spec["split"], dp["delta"],
                                     sigma_hist=dp["hist_noise_multiplier"])
    m = FairDPCTGAN(fair_train_cols=[], target_epsilon=dp["primary_epsilon"],
                    target_delta=dp["delta"], clip_norm=dp["clip_norm"],
                    hist_noise_multiplier=dp["hist_noise_multiplier"],
                    noise_multiplier=sig, epochs=cfg["ctgan"]["epochs_dp"],
                    seed=seed + 1000 * spec["rep"], **kw
                    ).fit(d["fit"], d["discrete"])
    runs = [budget.run(q, m.sigma_, steps, dp["hist_noise_multiplier"])]
    return m, runs


def fit_ddpm(cfg, d, spec, seed, device):
    t = cfg["tabddpm"]
    hidden = tuple(t["capacity"][spec["cap"]])
    common = dict(hidden=hidden, emb=t["emb"], timesteps=t["timesteps"],
                  device=device, verbose=False)
    if not spec["dp"]:
        m = TabDDPM(steps=t["steps"], batch_size=t["batch_size"], lr=t["lr"],
                    seed=seed, **common).fit(d["fit"], d["discrete"])
        return m, []
    dp = cfg["dp"]
    unit = spec.get("unit", "stay")
    m = DPTabDDPM(target_epsilon=dp["primary_epsilon"],
                  target_delta=dp["delta"], clip_norm=dp["clip_norm"],
                  epochs=t["dp_epochs"], expected_batch=t["dp_expected_batch"],
                  noise_multiplicity=t["noise_multiplicity"], unit=unit,
                  chunk=t.get("dp_chunk", 512), lr=t["dp_lr"],
                  seed=seed + 1000 * spec["rep"], **common)
    if spec.get("split"):
        n_units = (len(set(d["fit_groups"])) if unit == "patient"
                   else len(d["fit"]))
        q, steps, _ = m.plan_budget(n_units)
        m.noise_multiplier = budget.calibrate_split(
            dp["primary_epsilon"], q, steps, spec["split"], dp["delta"])
    m.fit(d["fit"], d["discrete"], groups=d["fit_groups"])
    runs = [budget.run(m.q_, m.sigma_, m.steps_)]
    return m, runs


# -------------------------------------------------------------- scoring ----
def score(task, cfg, d, syn, seed):
    tgt, disc = d["target"], d["discrete"]
    sens = cfg["fairness"]["sensitive"][task]
    dc, num = _cols(d["train"], tgt, disc)
    mia_disc = [c for c in disc if c != tgt]
    nc = cfg["mia"]["n_candidates"]
    return {
        "utility": SCORERS[task](syn, d["test"], tgt, dc, num, seed, sens),
        "mia": run_mia(d["train"], d["test"], syn, mia_disc, n_candidates=nc,
                       seed=seed),
        "mia_tail": mia_tail(d["train"], d["test"], syn, mia_disc,
                             n_candidates=nc, seed=seed),
        "fidelity": fidelity(d["train"], d["test"], syn, disc, tgt, seed=seed),
        "representation": representation(syn, d["train"], sens),
    }


def privacy_summary(cfg, runs):
    if not runs:
        return {"epsilon": None, "epsilon_add_remove": None, "runs": []}
    delta = cfg["dp"]["delta"]
    eps = budget.compose(runs, delta)
    # add/remove-one view: the gradient noise was calibrated to sensitivity
    # 2C, so against add/remove adjacency (sensitivity C) it is sigma_eff =
    # 2 sigma; exact for Poisson sampling (DP-TabDDPM), the same documented
    # approximation as Stage 1 for DP-CTGAN's fixed-size batches. Stage 1's
    # count release was already costed at add/remove sensitivity, so its
    # multiplier is unchanged.
    ar = budget.compose([dict(r, sigma=2 * r["sigma"]) for r in runs], delta)
    return {"epsilon": eps, "epsilon_add_remove": ar, "runs": runs}


# ----------------------------------------------------------------- driver --
def run_job(task, seed, arm, device="cpu", threads=None, cfg=None):
    cfg = dict(cfg or settings.load())
    cfg["seed"] = seed
    if threads:
        torch.set_num_threads(int(threads))
    out_p = result_path(task, seed, arm)
    if os.path.exists(out_p):
        print(f"[{task} {seed} {arm}] already done", flush=True)
        return
    t0 = time.time()
    d = RD.load_with_groups(task, cfg)
    tgt, disc = d["target"], d["discrete"]
    n = len(d["train"])
    res = {"task": task, "seed": seed, "arm": arm, "n_train": n,
           "n_test": len(d["test"]), "n_fit": len(d["fit"])}

    if arm == "TRTR":
        r = score(task, cfg, d, d["train"], seed)
        # the real-vs-real control: real training rows judged against real
        # test rows; the realism ceiling every generator is measured against
        res.update(r, privacy=privacy_summary(cfg, []))

    elif arm in GEN:
        spec = GEN[arm]
        dpath = data_path(cfg, task, seed, arm)
        meta_p = dpath.replace(".parquet", ".privacy.json")
        fair_p = data_path(cfg, task, seed, arm + "_rake")
        have = os.path.exists(dpath) and os.path.exists(meta_p) and (
            not spec.get("rake") or os.path.exists(fair_p))
        if have:
            meta = json.load(open(meta_p))
            runs, fit_s = meta["runs"], meta["fit_seconds"]
            syn = pd.read_parquet(dpath)
        else:
            tf = time.time()
            if spec["family"] == "ctgan":
                model, runs = fit_ctgan(cfg, d, spec, seed)
            else:
                model, runs = fit_ddpm(cfg, d, spec, seed, device)
            fit_s = time.time() - tf
            syn = _restore(model.sample(n), d["train"], disc)
            meta = {"runs": runs, "fit_seconds": fit_s}
            if getattr(model, "epsilon_add_remove_", None) is not None:
                meta["epsilon_add_remove_exact"] = model.epsilon_add_remove_
            if spec.get("rake"):
                # fair resampling of the same private generator: a pool is
                # drawn and raked to fair subgroup targets, zero extra epsilon
                sens = cfg["fairness"]["sensitive"][task]
                mult = 5 if len(sens) > 1 else 2
                pool = (model.sample(mult * n) if spec["family"] == "ctgan"
                        else model.sample(mult * n, seed_offset=11))
                pool = _restore(pool, d["train"], disc)
                fair, diag = fair_resample(pool, n, sens,
                                           cfg["fairness"]["min_support"],
                                           seed=seed)
                _save(fair, fair_p)
                _write({"runs": runs, "fit_seconds": 0.0, "parent": arm,
                        "resample": diag},
                       fair_p.replace(".parquet", ".privacy.json"))
            _save(syn, dpath)
            _write(meta, meta_p)
        syn = _restore(syn, d["train"], disc)
        res.update(score(task, cfg, d, syn, seed),
                   privacy=privacy_summary(cfg, runs), fit_seconds=fit_s,
                   device=device)
        if "epsilon_add_remove_exact" in meta:
            res["privacy"]["epsilon_add_remove_exact"] = \
                meta["epsilon_add_remove_exact"]

    elif arm in DERIVED:
        fpath = data_path(cfg, task, seed, arm)
        meta = json.load(open(fpath.replace(".parquet", ".privacy.json")))
        syn = _restore(pd.read_parquet(fpath), d["train"], disc)
        res.update(score(task, cfg, d, syn, seed),
                   privacy=privacy_summary(cfg, meta["runs"]),
                   parent=meta["parent"], resample=meta["resample"])

    elif arm in POOLED:
        comps = POOLED[arm]
        per = [n // len(comps) + (1 if i < n % len(comps) else 0)
               for i in range(len(comps))]
        frames, runs = [], []
        for c, k in zip(comps, per):
            frames.append(pd.read_parquet(data_path(cfg, task, seed, c))[:k])
            runs += json.load(open(data_path(cfg, task, seed, c).replace(
                ".parquet", ".privacy.json")))["runs"]
        rng = np.random.default_rng(seed)
        syn = pd.concat(frames, ignore_index=True)
        syn = syn.iloc[rng.permutation(len(syn))].reset_index(drop=True)
        syn = _restore(syn, d["train"], disc)
        res.update(score(task, cfg, d, syn, seed),
                   privacy=privacy_summary(cfg, runs), components=comps)
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    res["wall_seconds"] = time.time() - t0
    _write(res, out_p)
    u = res["utility"]
    key = "r2" if task == "los" else "roc_auc"
    eps = res["privacy"]["epsilon"]
    print(f"[{task} {seed} {arm}] {key}={u[key]:.3f} "
          f"det={res['fidelity']['detection_auc']:.3f} "
          f"eps={'inf' if eps is None else f'{eps:.2f}'} "
          f"({res['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    run_job(a.task, a.seed, a.arm, a.device or device_of(a.arm), a.threads)
