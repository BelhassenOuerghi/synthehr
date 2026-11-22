"""
generate.py
-----------
Fit the chosen generator on the REAL training data only and emit a synthetic
dataset of equal size for each task. Synthetic files are written OUTSIDE the
repository (to `synthetic_root`) because a synthetic set that has not yet passed
the privacy pillar is treated with the same caution as real data.

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

import data as D
from ctgan import CTGAN
import beddemand_gen as BD

warnings.filterwarnings("ignore")


def _fit_sample_tabular(ds, cfg):
    c = cfg["ctgan"]
    train = D.clean_numeric(ds["train"], ds["discrete"])
    fit_df = train
    if len(train) > c["max_train_rows"]:
        fit_df = train.sample(c["max_train_rows"], random_state=cfg["seed"]
                              ).reset_index(drop=True)
    print(f"[{ds['name']}] fitting CTGAN on {len(fit_df):,} rows, "
          f"{fit_df.shape[1]} cols ({len(ds['discrete'])} discrete)...", flush=True)
    t = time.time()
    model = CTGAN(
        embedding_dim=c["embedding_dim"], generator_dim=tuple(c["generator_dim"]),
        discriminator_dim=tuple(c["discriminator_dim"]), batch_size=c["batch_size"],
        epochs=c["epochs"], pac=c["pac"], gp_lambda=c["gp_lambda"],
        vgm_components=c["vgm_components"], d_steps=c["d_steps"],
        seed=cfg["seed"], verbose=True,
    ).fit(fit_df, ds["discrete"])
    print(f"[{ds['name']}] fit in {time.time()-t:.0f}s; sampling {len(train):,} rows",
          flush=True)
    syn = model.sample(len(train))
    # restore dtypes to match the real frame
    for col in train.columns:
        if col not in ds["discrete"]:
            syn[col] = pd.to_numeric(syn[col], errors="coerce")
    return syn[train.columns]


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    out = cfg["synthetic_root"]
    os.makedirs(out, exist_ok=True)
    manifest = {"generator": "CTGAN (self-implemented)", "seed": cfg["seed"],
                "ctgan": cfg["ctgan"], "datasets": {}}

    # --- tabular tasks -----------------------------------------------------
    for loader in (D.load_los, D.load_readmission):
        ds = loader(cfg)
        syn = _fit_sample_tabular(ds, cfg)
        path = os.path.join(out, f"{ds['name']}_synthetic.parquet")
        syn.to_parquet(path, index=False)
        manifest["datasets"][ds["name"]] = {
            "task": ds["task"], "target": ds["target"],
            "n_synthetic": len(syn), "n_real_train": len(ds["train"]),
            "n_real_test": len(ds["test"]), "path": path,
        }
        print(f"[{ds['name']}] wrote {len(syn):,} synthetic rows -> {path}", flush=True)

    # --- temporal task (Poisson-AR generative counterpart) -----------------
    bd = D.load_beddemand(cfg)
    W = cfg["beddemand"]["ar_window"]
    beta = BD.fit_poisson_ar(bd["y_train"], W)
    rng = np.random.default_rng(cfg["seed"])
    syn_series = BD.sample_series(beta, bd["y_train"][:W], len(bd["y_train"]),
                                  W, rng)
    bd_path = os.path.join(out, "beddemand_synthetic.csv")
    pd.DataFrame({"count": syn_series.astype(int)}).to_csv(bd_path, index=False)
    manifest["datasets"]["beddemand"] = {
        "task": "forecast", "generator": "Poisson-AR(%d) sampler" % W,
        "n_synthetic": int(len(syn_series)), "path": bd_path,
        "real_train_mean": float(np.mean(bd["y_train"])),
        "synthetic_mean": float(np.mean(syn_series)),
    }
    print(f"[beddemand] wrote {len(syn_series):,}-day synthetic series -> {bd_path}",
          flush=True)

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("generation complete; manifest written.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
