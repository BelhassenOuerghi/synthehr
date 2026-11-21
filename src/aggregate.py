"""
aggregate.py
------------
Turn the per-arm, per-seed results into the pre-registered analysis
(doc/PREREGISTRATION.md): means with 95% t-intervals, the primary paired tests
with Holm adjustment, each family's noise floor, spread (predictability) tests,
the capacity x privacy interaction, the fairness degeneracy check, and a
reproduction check of the CTGAN arms against the evidence repository.

Writes results/metrics_ci.json and results/results_table.md.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

import jobs as J
import settings

HERE = settings.HERE
RES = os.environ.get("ENGD_RESULTS_DIR", os.path.join(HERE, "results"))
EVIDENCE = os.path.join(os.path.dirname(HERE), "mimic-synthetic-evidence",
                        "results", "seeds")
KEY = {"los": "r2", "readmission": "roc_auc"}
UNIT = {"los": "R²", "readmission": "ROC-AUC"}
FID = ["detection_auc", "detection_auc_rounded", "precision", "recall",
       "marginal_w1", "marginal_tvd", "corr_error", "target_assoc_error"]
MIA = [("mia", "attack_auc_dcr"), ("mia", "attack_auc_knn_classifier"),
       ("mia_tail", "tpr@fpr=0.01"), ("mia_tail", "tpr@fpr=0.001")]


def load():
    out = {}
    for p in glob.glob(os.path.join(RES, "arms", "*", "seed_*", "*.json")):
        r = json.load(open(p))
        out.setdefault(r["task"], {}).setdefault(r["arm"], {})[r["seed"]] = r
    return out


def ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return None
    m = float(v.mean())
    if len(v) < 2:
        return {"mean": m, "half": 0.0, "sd": 0.0, "n": 1,
                "values": v.tolist()}
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))
    return {"mean": m, "half": half, "sd": sd, "n": int(len(v)),
            "values": v.tolist()}


def mde(diffs):
    """Minimum detectable paired effect at alpha = 0.05 and power 0.8, the
    evidence repository's definition: (t_.975 + t_.80) * sd(d) / sqrt(n)."""
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2:
        return None
    df = n - 1
    return float((stats.t.ppf(0.975, df) + stats.t.ppf(0.8, df))
                 * d.std(ddof=1) / np.sqrt(n))


def get(arm_runs, seed, path):
    r = arm_runs.get(seed)
    for k in path:
        if r is None:
            return None
        r = r.get(k) if isinstance(r, dict) else None
    return r


def series(data, task, arm, path, seeds):
    return [get(data[task].get(arm, {}), s, path) for s in seeds]


def paired(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x, y = np.array(pairs).T
    t, p = stats.ttest_rel(x, y)
    return {"mean_diff": float((x - y).mean()), "p": float(p),
            "n": len(pairs), "diffs": (x - y).tolist()}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj.tolist()


def noise_floor(data, task, family, seeds):
    base = "dpctgan" if family == "ctgan" else "dpddpm"
    key = ("utility", KEY[task])
    a = series(data, task, base, key, seeds)
    diffs, all_abs = {}, []
    for r in range(1, J.K):
        b = series(data, task, f"{base}_r{r}", key, seeds)
        d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
        diffs[r] = d
        all_abs += [abs(x) for x in d]
    reps = [mde(d) for d in diffs.values() if len(d) >= 2]
    within = []
    for s in seeds:
        vals = [get(data[task].get(x, {}), s, key)
                for x in [base] + [f"{base}_r{r}" for r in range(1, J.K)]]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            within.append(np.var(vals, ddof=1))
    return {"mean_abs_replicate_diff": float(np.mean(all_abs))
            if all_abs else None,
            "max_abs_replicate_diff": float(np.max(all_abs))
            if all_abs else None,
            "mde_n5": float(np.mean(reps)) if reps else None,
            "within_seed_sd": float(np.sqrt(np.mean(within)))
            if within else None}


def single_runs(data, task, base, seeds):
    key = ("utility", KEY[task])
    out = []
    for x in [base] + [f"{base}_r{r}" for r in range(1, J.K)]:
        out += [v for v in series(data, task, x, key, seeds) if v is not None]
    return out


def brown_forsythe(a, b):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    st, p = stats.levene(a, b, center="median")
    return {"sd_a": float(np.std(a, ddof=1)), "sd_b": float(np.std(b, ddof=1)),
            "n_a": len(a), "n_b": len(b), "p": float(p)}


def degenerate(arm_runs, seeds, cols):
    """Share of (seed, subgroup) cells whose positive-prediction rate is
    outside [2%, 98%], i.e. the classifier predicts (almost) one class."""
    bad = tot = 0
    for s in seeds:
        fair = get(arm_runs, s, ("utility", "fairness")) or {}
        for c in cols:
            for g in (fair.get(c, {}).get("per_group") or {}).values():
                if g.get("pos_rate") is None:
                    continue
                tot += 1
                bad += not (0.02 <= g["pos_rate"] <= 0.98)
    return {"degenerate_cells": bad, "cells": tot}


def reproduction(data, task, seeds):
    out = {}
    for mine, theirs in (("ctgan", "stage0"), ("dpctgan", "stage1"),
                         ("TRTR", "TRTR")):
        rows = []
        for s in seeds:
            p = os.path.join(EVIDENCE, f"seed_{s}.json")
            v = get(data[task].get(mine, {}), s, ("utility", KEY[task]))
            if v is None or not os.path.exists(p):
                continue
            ref = json.load(open(p))["tasks"][task]["arms"][theirs][KEY[task]]
            rows.append({"seed": s, "here": v, "evidence": ref,
                         "abs_diff": abs(v - ref)})
        out[mine] = rows
    return out


def main():
    data = load()
    cfg = settings.load()
    seeds = cfg["seeds"]
    agg = {"seeds": seeds, "tasks": {}, "primary": [], "secondary": []}
    prim = []
    for task in sorted(data):
        T = {"arms": {}}
        for arm, runs in sorted(data[task].items()):
            A = {"utility": {}, "fidelity": {}, "mia": {}, "privacy": {}}
            for k in ("r2", "rmse", "mae", "roc_auc", "pr_auc", "accuracy"):
                v = series(data, task, arm, ("utility", k), seeds)
                if any(x is not None for x in v):
                    A["utility"][k] = ci(v)
            for k in FID:
                A["fidelity"][k] = ci(series(data, task, arm,
                                             ("fidelity", k), seeds))
            for grp, k in MIA:
                A["mia"][k] = ci(series(data, task, arm, (grp, k), seeds))
            for k in ("epsilon", "epsilon_add_remove"):
                A["privacy"][k] = ci(series(data, task, arm,
                                            ("privacy", k), seeds))
            sens = cfg["fairness"]["sensitive"][task]
            A["fairness"] = {}
            for c in sens:
                for m in (("auc_gap", "dp_diff", "eo_diff")
                          if task == "readmission"
                          else ("rmse_gap", "bias_gap")):
                    A["fairness"][f"{c}.{m}"] = ci(series(
                        data, task, arm, ("utility", "fairness", c, m), seeds))
            A["degeneracy"] = degenerate(runs, seeds, sens)
            if task == "readmission":
                A["medicare_share"] = ci(series(
                    data, task, arm, ("representation", "insurance",
                                      "Medicare", "dataset"), seeds))
            T["arms"][arm] = A

        key = ("utility", KEY[task])

        def cmp(name, a, b, primary=True, path=key):
            r = paired(series(data, task, a, path, seeds),
                       series(data, task, b, path, seeds))
            if r is None:
                return
            r.update(task=task, name=name, a=a, b=b, metric=path[-1])
            (prim if primary else agg["secondary"]).append(r)

        cmp("RQ1 realism without privacy: TabDDPM vs CTGAN", "ddpm", "ctgan")
        cmp("RQ2 realism under privacy: DP-TabDDPM vs DP-CTGAN",
            "dpddpm", "dpctgan")
        cmp("RQ3b split-budget pool vs single run (CTGAN)",
            "dpctgan_ens4split", "dpctgan")
        cmp("RQ3b split-budget pool vs single run (TabDDPM)",
            "dpddpm_ens4split", "dpddpm")
        cmp("RQ5 cost of fair resampling (TabDDPM)", "dpddpm_rake", "dpddpm")
        cmp("RQ6 patient-level vs stay-level DP (TabDDPM)",
            "dpddpm_patient", "dpddpm")
        # RQ4: capacity x privacy interaction, a per-seed difference-in-
        # differences tested against zero
        dd = []
        for s in seeds:
            v = [get(data[task].get(x, {}), s, key)
                 for x in ("ddpm_L", "ddpm_S", "dpddpm_L", "dpddpm_S")]
            if None not in v:
                dd.append((v[0] - v[1]) - (v[2] - v[3]))
        if len(dd) >= 2:
            t, p = stats.ttest_1samp(dd, 0.0)
            prim.append({"task": task, "name": "RQ4 capacity x privacy "
                         "interaction: (L-S) without DP minus (L-S) with DP",
                         "a": "(ddpm_L-ddpm_S)", "b": "(dpddpm_L-dpddpm_S)",
                         "metric": KEY[task], "mean_diff": float(np.mean(dd)),
                         "p": float(p), "n": len(dd), "diffs": dd})
        # secondary: realism, privacy and the remaining contrasts
        for m in ("detection_auc", "precision", "recall", "target_assoc_error"):
            cmp(f"realism ({m}): TabDDPM vs CTGAN", "ddpm", "ctgan", False,
                ("fidelity", m))
            cmp(f"realism ({m}): DP-TabDDPM vs DP-CTGAN", "dpddpm", "dpctgan",
                False, ("fidelity", m))
        cmp("cost of synthesis (TRTR vs TabDDPM)", "TRTR", "ddpm", False)
        cmp("cost of privacy (TabDDPM vs DP-TabDDPM)", "ddpm", "dpddpm",
            False)
        cmp("cost of privacy (CTGAN vs DP-CTGAN)", "ctgan", "dpctgan", False)
        cmp("RQ3a full-budget pool vs single run (CTGAN)", "dpctgan_ens4",
            "dpctgan", False)
        cmp("RQ3a full-budget pool vs single run (TabDDPM)", "dpddpm_ens4",
            "dpddpm", False)
        cmp("cost of fair resampling (CTGAN)", "dpctgan_rake", "dpctgan",
            False)
        for cap in ("S", "L"):
            cmp(f"capacity {cap} vs M, non-private", f"ddpm_{cap}", "ddpm",
                False)
            cmp(f"capacity {cap} vs M, DP", f"dpddpm_{cap}", "dpddpm", False)

        T["noise_floor"] = {f: noise_floor(data, task, f, seeds)
                            for f in ("ctgan", "ddpm")}
        T["spread"] = {}
        for fam, base in (("ctgan", "dpctgan"), ("ddpm", "dpddpm")):
            singles = single_runs(data, task, base, seeds)
            for pool in ("ens4", "ens4split"):
                T["spread"][f"{base} single runs vs {base}_{pool}"] = \
                    brown_forsythe(singles,
                                   series(data, task, f"{base}_{pool}", key,
                                          seeds))
            T["spread"][f"{fam}: non-private vs DP (seed sd)"] = \
                brown_forsythe(series(data, task,
                                      "ctgan" if fam == "ctgan" else "ddpm",
                                      key, seeds),
                               series(data, task, base, key, seeds))
        T["reproduction"] = reproduction(data, task, seeds)
        agg["tasks"][task] = T

    if prim:
        adj = holm([r["p"] for r in prim])
        for r, a in zip(prim, adj):
            r["p_holm"] = a
            nf = agg["tasks"][r["task"]]["noise_floor"][
                "ctgan" if "CTGAN" in r["name"] and "TabDDPM" not in r["name"]
                else "ddpm"]["mean_abs_replicate_diff"]
            r["noise_floor"] = nf
            r["verdict"] = ("SIGNIFICANT" if r["p"] < 0.05 else
                            "within noise")
            if r["p"] < 0.05 and nf is not None and abs(r["mean_diff"]) < nf:
                r["verdict"] = "significant, below noise floor"
    agg["primary"] = prim
    with open(os.path.join(RES, "metrics_ci.json"), "w") as f:
        json.dump(agg, f, indent=2)
    table(agg)
    print(f"aggregated {sum(len(v) for v in data.values())} arms over "
          f"{len(data)} tasks -> results/metrics_ci.json, results_table.md")


def f(c, d=3):
    if not c:
        return "–"
    return f"{c['mean']:.{d}f} ± {c['half']:.{d}f}"


ORDER = ["TRTR", "ctgan", "ddpm", "ddpm_S", "ddpm_L", "dpctgan", "dpctgan_r1",
         "dpctgan_r2", "dpctgan_r3", "dpctgan_ens4", "dpctgan_s1",
         "dpctgan_ens4split", "dpctgan_rake", "dpddpm", "dpddpm_r1",
         "dpddpm_r2", "dpddpm_r3", "dpddpm_ens4", "dpddpm_s1",
         "dpddpm_ens4split", "dpddpm_rake", "dpddpm_patient", "dpddpm_S",
         "dpddpm_L"]


def table(a):
    L = ["# Pillar 4 (realism): results", "",
         f"Seeds {a['seeds']}, patient-disjoint splits identical to the "
         "evidence repository. Mean ± 95% CI over seeds. ε is in Stage 1's "
         "convention (ε_ar: the same runs under add/remove adjacency).", ""]
    for task, T in a["tasks"].items():
        u = UNIT[task]
        L += [f"## {task}: utility, realism and privacy", "",
              f"| Arm | {u} | detection AUC ↓ | precision ↑ | recall ↑ | "
              "target-assoc. err ↓ | DCR attack | TPR@1%FPR | ε | ε_ar |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ORDER:
            A = T["arms"].get(arm)
            if not A:
                continue
            k = KEY[task]
            L += [f"| {arm} | {f(A['utility'].get(k))} "
                  f"| {f(A['fidelity']['detection_auc'])} "
                  f"| {f(A['fidelity']['precision'])} "
                  f"| {f(A['fidelity']['recall'])} "
                  f"| {f(A['fidelity']['target_assoc_error'])} "
                  f"| {f(A['mia']['attack_auc_dcr'])} "
                  f"| {f(A['mia']['tpr@fpr=0.01'])} "
                  f"| {f(A['privacy']['epsilon'], 2)} "
                  f"| {f(A['privacy']['epsilon_add_remove'], 2)} |"]
        nf = T["noise_floor"]

        def g(x):
            return "–" if x is None else f"{x:.3f}"

        L += ["", f"**Noise floors ({u}).** " + "; ".join(
            f"DP-{fam.upper() if fam == 'ctgan' else 'TabDDPM'}: mean "
            f"|replicate diff| {g(v['mean_abs_replicate_diff'])}, max "
            f"{g(v['max_abs_replicate_diff'])}, MDE(n=5) {g(v['mde_n5'])}, "
            f"within-seed sd {g(v['within_seed_sd'])}"
            for fam, v in nf.items() if v["mean_abs_replicate_diff"]
            is not None), ""]
        L += ["**Spread (Brown-Forsythe).**", "",
              "| Comparison | sd A | sd B | n A | n B | p |",
              "|---|---|---|---|---|---|"]
        for name, r in T["spread"].items():
            if r:
                L += [f"| {name} | {r['sd_a']:.3f} | {r['sd_b']:.3f} "
                      f"| {r['n_a']} | {r['n_b']} | {r['p']:.3f} |"]
        L += ["", "**Reproduction of the evidence repository** "
              "(same seed, same recipe): " + "; ".join(
                  f"{arm}: max |Δ| = "
                  f"{max(x['abs_diff'] for x in rows):.4f} over "
                  f"{len(rows)} seeds" for arm, rows in
                  T["reproduction"].items() if rows), ""]
        if task == "readmission":
            L += ["**Fairness at restored utility** (readmission, "
                  "insurance):", "",
                  "| Arm | AUC gap | DP diff | EO diff | degenerate cells | "
                  "Medicare share |", "|---|---|---|---|---|---|"]
            for arm in ("TRTR", "ctgan", "ddpm", "dpctgan", "dpctgan_rake",
                        "dpddpm", "dpddpm_rake"):
                A = T["arms"].get(arm)
                if not A:
                    continue
                dg = A["degeneracy"]
                L += [f"| {arm} | {f(A['fairness'].get('insurance.auc_gap'))} "
                      f"| {f(A['fairness'].get('insurance.dp_diff'))} "
                      f"| {f(A['fairness'].get('insurance.eo_diff'))} "
                      f"| {dg['degenerate_cells']}/{dg['cells']} "
                      f"| {f(A.get('medicare_share'))} |"]
            L += [""]
    L += ["## Pre-registered primary comparisons (paired t-test, Holm)", "",
          "| Task | Comparison | Δ | p | p (Holm) | noise floor | verdict |",
          "|---|---|---|---|---|---|---|"]
    for r in a["primary"]:
        nf = r.get("noise_floor")
        L += [f"| {r['task']} | {r['name']} | {r['mean_diff']:+.3f} "
              f"| {r['p']:.4f} | {r['p_holm']:.4f} "
              f"| {'–' if nf is None else f'{nf:.3f}'} | {r['verdict']} |"]
    L += ["", "## Secondary comparisons (paired t-test, unadjusted)", "",
          "| Task | Comparison | metric | Δ | p |", "|---|---|---|---|---|"]
    for r in a["secondary"]:
        L += [f"| {r['task']} | {r['name']} | {r['metric']} "
              f"| {r['mean_diff']:+.3f} | {r['p']:.4f} |"]
    with open(os.path.join(RES, "results_table.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
