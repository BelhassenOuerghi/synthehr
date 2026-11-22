"""
backtest.py
-----------
Rolling-origin backtest of 14-day-ahead forecasts, mirroring the upstream
evaluation protocol (Lee et al. 2021: F = 14, heldout log-likelihood per day)
and extending it with point-forecast and calibration metrics.

Protocol per target series (admissions, census):
  1. AR window size W is selected on a 52-week validation slice by one-step
     heldout log-likelihood (analogue of upstream's gar_grid_search).
  2. The final 52 weeks are the test span. 26 forecast origins tile it at a
     14-day stride; at each origin the models are refit on all prior data
     (expanding window) and 2000 Monte Carlo trajectories are simulated.
  3. Metrics: MAE (of predictive median), RMSE (of predictive mean),
     95%/50% interval coverage and width, upstream-style heldout log-lik per
     day (joint, log-mean-exp over trajectories), exact one-step log-lik on
     the whole test span, and PIT calibration error (MACE, 10 bins).

Outputs: results/metrics.json, results/results_table.md, and a git-ignored
plotting payload data/derived/backtest_payload.npz.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from genpoisson import fit_gp_iid, fit_poisson_iid
from models import CountARModel, SeasonalNaive, MovingAverage


def load_series(cfg, target):
    path = os.path.join(cfg["derived_dir"], f"{target}_daily.csv")
    df = pd.read_csv(path, parse_dates=["day"])
    return df["day"].to_numpy(), df["count"].to_numpy(dtype=float)


def select_window(y, val_start, test_start, grid, genpoisson):
    """Pick W maximizing mean one-step log-lik on the validation slice."""
    best_w, best_ll = None, -np.inf
    for W in grid:
        m = CountARModel(window=W, genpoisson=genpoisson).fit(y[:val_start])
        lls = m.loglik_per_obs(y[:test_start])          # entries for t = W..test_start-1
        val_ll = float(np.mean(lls[-(test_start - val_start):]))
        print(f"    W={W}: val one-step loglik {val_ll:.4f}")
        if val_ll > best_ll:
            best_w, best_ll = W, val_ll
    return best_w, best_ll


def evaluate_target(cfg, target, rng):
    dates, y = load_series(cfg, target)
    N = len(y)
    F = cfg["horizon"]
    S = cfg["n_forecast_samples"]
    stride = cfg["origin_stride"]
    test_start = N - cfg["test_days"]
    val_start = test_start - cfg["val_days"]
    origins = list(range(test_start, N - F + 1, stride))
    pu = bool(cfg["param_uncertainty"])

    print(f"\n=== target: {target}, N={N}, train<{val_start}, "
          f"val[{val_start}:{test_start}], test[{test_start}:{N}], "
          f"{len(origins)} origins ===")

    out = {"series": {
        "n_days": int(N),
        "start": str(pd.Timestamp(dates[0]).date()),
        "end": str(pd.Timestamp(dates[-1]).date()),
        "mean": float(y.mean()),
        "variance": float(y.var(ddof=1)),
        "dispersion_ratio_var_over_mean": float(y.var(ddof=1) / y.mean()),
        "zero_days_fraction": float((y == 0).mean()),
        "max": int(y.max()),
    }}

    # --- marginal (iid) dispersion diagnostic on pre-test data --------------
    y_pre = y[:test_start]
    mu_p, ll_p = fit_poisson_iid(y_pre)
    th_g, lam_g, ll_g = fit_gp_iid(y_pre)
    out["marginal_fit_pre_test"] = {
        "poisson": {"mu": mu_p, "loglik_per_day": ll_p / len(y_pre)},
        "genpoisson": {"theta": th_g, "lambda": lam_g,
                       "implied_dispersion": 1.0 / (1.0 - lam_g) ** 2,
                       "loglik_per_day": ll_g / len(y_pre)},
    }

    # --- window selection ----------------------------------------------------
    grid = cfg["ar_window_grid"]
    print("  window selection (GenPoisson):")
    Wg, _ = select_window(y, val_start, test_start, grid, genpoisson=True)
    print("  window selection (Poisson):")
    Wp, _ = select_window(y, val_start, test_start, grid, genpoisson=False)
    out["selected_window"] = {"gp_ar": Wg, "poisson_ar": Wp}
    print(f"  selected W: GP-AR={Wg}, Poisson-AR={Wp}")

    # --- exact one-step scores + PIT on the test span (fit on pre-test) -----
    gp_fixed = CountARModel(window=Wg, genpoisson=True).fit(y_pre)
    po_fixed = CountARModel(window=Wp, genpoisson=False).fit(y_pre)
    onestep = {}
    pits = {}
    onestep_theta = {}
    for name, m in [("gp_ar", gp_fixed), ("poisson_ar", po_fixed)]:
        lls = m.loglik_per_obs(y)[-(N - test_start):]
        onestep[name] = float(np.mean(lls))
        pits[name] = m.pit_one_step(y, rng)[-(N - test_start):]
        onestep_theta[name] = m.cond_theta(y)[0][-(N - test_start):]
    out["conditional_dispersion_pearson_poisson_ar"] = float(np.mean(
        (po_fixed.cond_theta(y_pre)[1] - po_fixed.cond_theta(y_pre)[0]) ** 2
        / po_fixed.cond_theta(y_pre)[0]))
    out["fixed_fit_params"] = {
        "gp_ar": {"beta": [float(b) for b in gp_fixed.beta],
                  "lambda": float(gp_fixed.lam),
                  "implied_conditional_dispersion": float(1.0 / (1.0 - gp_fixed.lam) ** 2)},
        "poisson_ar": {"beta": [float(b) for b in po_fixed.beta]},
    }

    # --- rolling-origin backtest ---------------------------------------------
    model_specs = {
        "gp_ar": lambda: CountARModel(window=Wg, genpoisson=True),
        "poisson_ar": lambda: CountARModel(window=Wp, genpoisson=False),
        "seasonal_naive": lambda: SeasonalNaive(),
        "moving_average": lambda: MovingAverage(cfg["baseline_ma_window"]),
    }
    rows = {k: {"med": [], "mean": [], "lo95": [], "hi95": [],
                "lo50": [], "hi50": [], "obs": [],
                "joint_ll": [], "rb_ll": []} for k in model_specs}
    payload_forecasts = {}

    for oi, origin in enumerate(origins):
        hist, obs = y[:origin], y[origin:origin + F]
        for name, make in model_specs.items():
            m = make()
            if isinstance(m, CountARModel):
                m.fit(hist)
                smp, thetas, lams = m.forecast_samples(hist, F, S, rng,
                                                       param_uncertainty=pu)
                jll, rb = m.heldout_loglik(thetas, lams, obs)
                rows[name]["joint_ll"].append(jll)
                rows[name]["rb_ll"].append(np.mean(rb))
            else:
                m.fit(hist, F)
                smp, _, _ = m.forecast_samples(hist, F, S, rng)
            r = rows[name]
            r["med"].append(np.percentile(smp, 50, axis=0))
            r["mean"].append(smp.mean(axis=0))
            r["lo95"].append(np.percentile(smp, 2.5, axis=0))
            r["hi95"].append(np.percentile(smp, 97.5, axis=0))
            r["lo50"].append(np.percentile(smp, 25, axis=0))
            r["hi50"].append(np.percentile(smp, 75, axis=0))
            r["obs"].append(obs)
            if oi == len(origins) - 1:                # signature-plot origin
                payload_forecasts[name] = smp
        print(f"  origin {oi + 1}/{len(origins)} done")

    # --- aggregate metrics ----------------------------------------------------
    metrics = {}
    for name, r in rows.items():
        obs = np.concatenate(r["obs"])
        med = np.concatenate(r["med"])
        mean = np.concatenate(r["mean"])
        lo95, hi95 = np.concatenate(r["lo95"]), np.concatenate(r["hi95"])
        lo50, hi50 = np.concatenate(r["lo50"]), np.concatenate(r["hi50"])
        m = {
            "mae_median": float(np.mean(np.abs(obs - med))),
            "rmse_mean": float(np.sqrt(np.mean((obs - mean) ** 2))),
            "coverage_95": float(np.mean((obs >= lo95) & (obs <= hi95))),
            "coverage_50": float(np.mean((obs >= lo50) & (obs <= hi50))),
            "mean_width_95": float(np.mean(hi95 - lo95)),
        }
        if name in ("gp_ar", "poisson_ar"):
            m["heldout_loglik_per_day_joint"] = float(np.mean(r["joint_ll"]))
            m["heldout_loglik_per_day_rb"] = float(np.mean(r["rb_ll"]))
            m["one_step_loglik_test"] = onestep[name]
            u = pits[name]
            hist_, _ = np.histogram(u, bins=10, range=(0, 1))
            m["pit_mace_10bin"] = float(np.mean(np.abs(hist_ / len(u) - 0.1)))
        metrics[name] = m
    out["models"] = metrics

    payload = {
        "dates": dates.astype("datetime64[D]").astype(str),
        "y": y,
        "test_start": test_start,
        "origins": np.array(origins),
        "sig_origin": origins[-1],
        "obs": {k: np.stack(v["obs"]) for k, v in rows.items()},
        "med": {k: np.stack(v["med"]) for k, v in rows.items()},
        "lo95": {k: np.stack(v["lo95"]) for k, v in rows.items()},
        "hi95": {k: np.stack(v["hi95"]) for k, v in rows.items()},
        "sig_samples": payload_forecasts,
        "pit": pits,
        "onestep_theta": onestep_theta,
        "onestep_lam": {"gp_ar": np.array(gp_fixed.lam),
                        "poisson_ar": np.array(0.0)},
        "marginal_fit": out["marginal_fit_pre_test"],
    }
    return out, payload


def write_results_table(all_out, path):
    lines = ["# Replication results: rolling 14-day-ahead backtest", ""]
    for target, out in all_out.items():
        s = out["series"]
        lines += [
            f"## Target: {target}",
            "",
            f"- window: {s['start']} to {s['end']} ({s['n_days']} days); "
            f"mean {s['mean']:.2f}, variance {s['variance']:.2f}, "
            f"**Var/Mean = {s['dispersion_ratio_var_over_mean']:.3f}** (marginal)",
            f"- conditional (Pearson) dispersion under Poisson-AR: "
            f"**{out['conditional_dispersion_pearson_poisson_ar']:.3f}**",
            f"- GP-AR dispersion parameter: lambda = "
            f"{out['fixed_fit_params']['gp_ar']['lambda']:.3f} "
            f"(implied conditional Var/Mean = "
            f"{out['fixed_fit_params']['gp_ar']['implied_conditional_dispersion']:.3f})",
            f"- selected AR window W: GP-AR = {out['selected_window']['gp_ar']}, "
            f"Poisson-AR = {out['selected_window']['poisson_ar']}",
            "",
            "| model | MAE | RMSE | 95% cov. | 50% cov. | 95% width | "
            "heldout ll/day (joint) | one-step ll | PIT MACE |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name in ["gp_ar", "poisson_ar", "seasonal_naive", "moving_average"]:
            m = out["models"][name]
            jll = f"{m['heldout_loglik_per_day_joint']:.4f}" if "heldout_loglik_per_day_joint" in m else ","
            osll = f"{m['one_step_loglik_test']:.4f}" if "one_step_loglik_test" in m else ","
            mace = f"{m['pit_mace_10bin']:.4f}" if "pit_mace_10bin" in m else ","
            lines.append(
                f"| {name} | {m['mae_median']:.3f} | {m['rmse_mean']:.3f} | "
                f"{m['coverage_95']:.3f} | {m['coverage_50']:.3f} | "
                f"{m['mean_width_95']:.2f} | {jll} | {osll} | {mace} |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["derived_dir"], exist_ok=True)

    all_out, payloads = {}, {}
    for target in cfg["targets"]:
        out, payload = evaluate_target(cfg, target, rng)
        all_out[target] = out
        payloads[target] = payload

    metrics = {
        "protocol": {
            "horizon_days": cfg["horizon"],
            "test_days": cfg["test_days"],
            "val_days": cfg["val_days"],
            "origin_stride": cfg["origin_stride"],
            "n_forecast_samples": cfg["n_forecast_samples"],
            "param_uncertainty": cfg["param_uncertainty"],
            "seed": cfg["seed"],
            "window": [cfg["window_start"], cfg["window_end"]],
        },
        "targets": all_out,
    }
    with open(os.path.join(cfg["results_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    write_results_table(all_out, os.path.join(cfg["results_dir"], "results_table.md"))

    np.savez_compressed(
        os.path.join(cfg["derived_dir"], "backtest_payload.npz"),
        **{f"{t}__{k}": v for t, p in payloads.items()
           for k, v in _flatten(p).items()})
    print("\nwrote results/metrics.json, results/results_table.md, "
          "data/derived/backtest_payload.npz")


def _flatten(payload):
    flat = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k == "marginal_fit":
                    continue
                flat[f"{k}__{k2}"] = v2
        else:
            flat[k] = v
    flat["marginal_fit_json"] = np.array(json.dumps(payload["marginal_fit"]))
    return flat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
