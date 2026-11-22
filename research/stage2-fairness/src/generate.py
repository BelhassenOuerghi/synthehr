"""
generate.py
-----------
Stage-2 generation: at the release budget (ε = 10, δ = 1e-5) fit the DP
generator per task and emit THREE arms per tabular task:

  * dpbaseline, standard DP-CTGAN training, standard sampling
                 (the Stage-1 recipe re-run as the within-stage control);
  * fairsample, the SAME trained generator, sampled with fair conditioning
                 on the sensitive columns (zero additional privacy cost);
  * fairtrain, a second training with uniform training-by-sampling on the
                 sensitive columns, then fair sampling.

Bed demand carries no sensitive attribute (a count series), so the Stage-1 DP
Poisson-AR is re-fit at the release budget and passed through, stated as a
scope decision. The canonical Stage-2 files copy the configured arm.

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
from fair_dp_ctgan import FairDPCTGAN

warnings.filterwarnings("ignore")


def _fit(ds, cfg, fair_train_cols, tag):
    c, dp = cfg["ctgan"], cfg["dp"]
    train = D.clean_numeric(ds["train"], ds["discrete"])
    fit_df = train
    if len(train) > c["max_train_rows"]:
        fit_df = train.sample(c["max_train_rows"], random_state=cfg["seed"]
                              ).reset_index(drop=True)
    print(f"[{ds['name']}|{tag}] fitting on {len(fit_df):,} rows "
          f"(fair_train_cols={fair_train_cols})...", flush=True)
    t = time.time()
    model = FairDPCTGAN(
        fair_train_cols=fair_train_cols,
        target_epsilon=dp["primary_epsilon"], target_delta=dp["delta"],
        clip_norm=dp["clip_norm"],
        hist_noise_multiplier=dp["hist_noise_multiplier"],
        embedding_dim=c["embedding_dim"],
        generator_dim=tuple(c["generator_dim"]),
        discriminator_dim=tuple(c["discriminator_dim"]),
        batch_size=c["batch_size"], epochs=c["epochs"], pac=c["pac"],
        gp_lambda=c["gp_lambda"], vgm_components=c["vgm_components"],
        d_steps=c["d_steps"], seed=cfg["seed"], verbose=True,
    ).fit(fit_df, ds["discrete"])
    print(f"[{ds['name']}|{tag}] fit in {time.time()-t:.0f}s "
          f"(eps spent {model.epsilon_spent_:.2f})", flush=True)
    return model, train


def _postprocess(syn, train, discrete):
    for col in train.columns:
        if col not in discrete:
            syn[col] = pd.to_numeric(syn[col], errors="coerce")
    return syn[train.columns]


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    out, dp, fair = cfg["synthetic_root"], cfg["dp"], cfg["fairness"]
    os.makedirs(out, exist_ok=True)
    manifest = {"generator": "Fair-DP-CTGAN (fair conditional generation on a "
                             "DP-SGD critic; RDP-accounted)",
                "seed": cfg["seed"], "ctgan": cfg["ctgan"], "dp": dp,
                "fairness": fair, "datasets": {}}

    for loader in (D.load_los, D.load_readmission):
        ds = loader(cfg)
        sens = fair["sensitive"][ds["name"]]
        entry = {"task": ds["task"], "target": ds["target"],
                 "sensitive": sens, "n_real_train": len(ds["train"]),
                 "arms": {}}

        # --- training A: standard DP; arms dpbaseline + fairsample ----------
        model, train = _fit(ds, cfg, [], "standard-train")
        plan = model.build_plan(sens, fair["min_support"])
        entry["noisy_shares"] = {c: model.noisy_shares_[c].round(4).tolist()
                                 for c in sens}
        entry["fair_targets"] = {c: p.round(4).tolist() for c, p in plan}
        for arm, sampler in (("dpbaseline", lambda: model.sample(len(train))),
                             ("fairsample",
                              lambda: model.sample_fair(len(train), plan))):
            syn = _postprocess(sampler(), train, ds["discrete"])
            path = os.path.join(out, f"{ds['name']}_synthetic_{arm}.parquet")
            syn.to_parquet(path, index=False)
            entry["arms"][arm] = {"path": path,
                                  "epsilon_spent": model.epsilon_spent_,
                                  "sigma": model.sigma_,
                                  "n_synthetic": len(syn)}
            print(f"[{ds['name']}|{arm}] wrote {len(syn):,} rows", flush=True)

        # --- training B: fair training-by-sampling; arm fairtrain -----------
        model_ft, _ = _fit(ds, cfg, sens, "fair-train")
        plan_ft = model_ft.build_plan(sens, fair["min_support"])
        syn = _postprocess(model_ft.sample_fair(len(train), plan_ft),
                           train, ds["discrete"])
        path = os.path.join(out, f"{ds['name']}_synthetic_fairtrain.parquet")
        syn.to_parquet(path, index=False)
        entry["arms"]["fairtrain"] = {"path": path,
                                      "epsilon_spent": model_ft.epsilon_spent_,
                                      "sigma": model_ft.sigma_,
                                      "n_synthetic": len(syn)}
        print(f"[{ds['name']}|fairtrain] wrote {len(syn):,} rows", flush=True)

        # --- canonical Stage-2 file -----------------------------------------
        canon = cfg["canonical_arm"]
        pd.read_parquet(entry["arms"][canon]["path"]).to_parquet(
            os.path.join(out, f"{ds['name']}_synthetic.parquet"), index=False)
        entry["canonical_arm"] = canon
        manifest["datasets"][ds["name"]] = entry

    # --- temporal pass-through (no sensitive attribute on a count series) ---
    bd = D.load_beddemand(cfg)
    W = cfg["beddemand"]["ar_window"]
    beta, eps_spent, sigma = dp_fit_poisson_ar(
        bd["y_train"], W, target_epsilon=dp["primary_epsilon"],
        target_delta=dp["delta_beddemand"], clip_norm=dp["clip_norm"],
        iters=dp["beddemand_iters"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    syn_series = BD.sample_series(beta, bd["y_train"][:W], len(bd["y_train"]),
                                  W, rng)
    bd_path = os.path.join(out, "beddemand_synthetic.csv")
    pd.DataFrame({"count": syn_series.astype(int)}).to_csv(bd_path, index=False)
    manifest["datasets"]["beddemand"] = {
        "task": "forecast", "generator": f"DP Poisson-AR({W}) pass-through",
        "note": "no sensitive attribute on a single count series (scope)",
        "epsilon_spent": eps_spent, "sigma": sigma, "path": bd_path,
    }
    print("[beddemand] wrote DP pass-through series", flush=True)

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("generation complete; manifest written.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
