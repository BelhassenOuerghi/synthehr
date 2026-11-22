"""
generate.py
-----------
Fit the DP generators on the REAL training data only and emit, for every ε in
the configured grid, a synthetic dataset of equal size per task. The primary ε
additionally writes the canonical (unsuffixed) Stage-1 files that the
downstream ladder consumes. Synthetic files are written OUTSIDE the repository
(to `synthetic_root`, i.e. MIMIC-SyntheticPrivacy).

    python src/generate.py [--config config.yaml]
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

import beddemand_gen as BD
import data as D
from dp_beddemand import dp_fit_poisson_ar
from dp_ctgan import DPCTGAN

warnings.filterwarnings("ignore")


def _eps_tag(eps):
    return f"eps{eps:g}".replace(".", "p")


def _fit_sample_tabular(ds, cfg, eps):
    c, dp = cfg["ctgan"], cfg["dp"]
    train = D.clean_numeric(ds["train"], ds["discrete"])
    fit_df = train
    if len(train) > c["max_train_rows"]:
        fit_df = train.sample(c["max_train_rows"], random_state=cfg["seed"]
                              ).reset_index(drop=True)
    print(f"[{ds['name']}|eps={eps}] fitting DP-CTGAN on {len(fit_df):,} rows, "
          f"{fit_df.shape[1]} cols ({len(ds['discrete'])} discrete)...",
          flush=True)
    t = time.time()
    model = DPCTGAN(
        target_epsilon=eps, target_delta=dp["delta"],
        clip_norm=dp["clip_norm"],
        hist_noise_multiplier=dp["hist_noise_multiplier"],
        embedding_dim=c["embedding_dim"],
        generator_dim=tuple(c["generator_dim"]),
        discriminator_dim=tuple(c["discriminator_dim"]),
        batch_size=c["batch_size"], epochs=c["epochs"], pac=c["pac"],
        gp_lambda=c["gp_lambda"], vgm_components=c["vgm_components"],
        d_steps=c["d_steps"], seed=cfg["seed"], verbose=True,
    ).fit(fit_df, ds["discrete"])
    print(f"[{ds['name']}|eps={eps}] fit in {time.time()-t:.0f}s "
          f"(eps spent {model.epsilon_spent_:.2f}); "
          f"sampling {len(train):,} rows", flush=True)
    syn = model.sample(len(train))
    for col in train.columns:
        if col not in ds["discrete"]:
            syn[col] = pd.to_numeric(syn[col], errors="coerce")
    return syn[train.columns], model


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    out, dp = cfg["synthetic_root"], cfg["dp"]
    os.makedirs(out, exist_ok=True)
    grid = list(dp["epsilon_grid"])
    primary = dp["primary_epsilon"]
    manifest = {"generator": "DP-CTGAN (self-implemented, DP-SGD critic + "
                             "RDP accountant)",
                "seed": cfg["seed"], "ctgan": cfg["ctgan"], "dp": dp,
                "datasets": {}}

    # --- tabular tasks ------------------------------------------------------
    for loader in (D.load_los, D.load_readmission):
        ds = loader(cfg)
        entries = {}
        for eps in grid:
            syn, model = _fit_sample_tabular(ds, cfg, eps)
            path = os.path.join(
                out, f"{ds['name']}_synthetic_{_eps_tag(eps)}.parquet")
            syn.to_parquet(path, index=False)
            if eps == primary:
                canon = os.path.join(out, f"{ds['name']}_synthetic.parquet")
                syn.to_parquet(canon, index=False)
            entries[str(eps)] = {
                "path": path, "epsilon_spent": model.epsilon_spent_,
                "sigma": model.sigma_, "n_synthetic": len(syn),
            }
            print(f"[{ds['name']}|eps={eps}] wrote {len(syn):,} rows -> {path}",
                  flush=True)
        manifest["datasets"][ds["name"]] = {
            "task": ds["task"], "target": ds["target"],
            "n_real_train": len(ds["train"]), "n_real_test": len(ds["test"]),
            "primary_epsilon": primary, "by_epsilon": entries,
        }

    # --- temporal task ------------------------------------------------------
    bd = D.load_beddemand(cfg)
    W = cfg["beddemand"]["ar_window"]
    entries = {}
    for eps in grid:
        beta, eps_spent, sigma = dp_fit_poisson_ar(
            bd["y_train"], W, target_epsilon=eps,
            target_delta=dp["delta_beddemand"], clip_norm=dp["clip_norm"],
            iters=dp["beddemand_iters"], seed=cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        syn_series = BD.sample_series(beta, bd["y_train"][:W],
                                      len(bd["y_train"]), W, rng)
        path = os.path.join(out, f"beddemand_synthetic_{_eps_tag(eps)}.csv")
        pd.DataFrame({"count": syn_series.astype(int)}).to_csv(path,
                                                               index=False)
        if eps == primary:
            pd.DataFrame({"count": syn_series.astype(int)}).to_csv(
                os.path.join(out, "beddemand_synthetic.csv"), index=False)
        entries[str(eps)] = {"path": path, "epsilon_spent": eps_spent,
                             "sigma": sigma,
                             "synthetic_mean": float(np.mean(syn_series))}
        print(f"[beddemand|eps={eps}] wrote series -> {path}", flush=True)
    manifest["datasets"]["beddemand"] = {
        "task": "forecast", "generator": f"DP Poisson-AR({W})",
        "real_train_mean": float(np.mean(bd["y_train"])),
        "primary_epsilon": primary, "by_epsilon": entries,
    }

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("generation complete; manifest written.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
