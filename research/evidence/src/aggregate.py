"""
aggregate.py
------------
Turn the per-seed results into the programme's headline evidence: mean ± 95%
confidence interval for every arm and metric, plus the decisive test,
**which between-arm differences actually survive seed noise.**

CIs use the t-distribution (small n), which is the honest choice for n = 5.
A difference is reported as SIGNIFICANT only if a paired t-test across seeds
(the arms share a seed, so pairing is valid and far more powerful) rejects at
alpha = 0.05. Everything else is reported as "within noise", which, for several
single-seed claims in Stages 1-2, is the correct and more honest verdict.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["TRTR", "stage0", "stage1", "dpbaseline", "fairsample", "fairtrain"]
HEADLINE = {"los": ["r2", "rmse", "mae"],
            "readmission": ["roc_auc", "pr_auc", "accuracy"]}


def ci(vals, alpha=0.05):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return m, 0.0, m, m
    half = stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)
    return m, float(half), m - half, m + half


def paired(a, b, alpha=0.05):
    """Paired t-test across seeds: is arm a different from arm b?"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return None
    t, p = stats.ttest_rel(a, b)
    return {"mean_diff": float((a - b).mean()), "p_value": float(p),
            "significant": bool(p < alpha)}


def _noise(seeds):
    """Independent DP replicates (same recipe/split, different model seed).
    Returns {task: [values...]} aligned to `seeds`, or None if not yet run."""
    out, key = {}, None
    for s in seeds:
        p = os.path.join(HERE, "results", "seeds", f"noise_{s}.json")
        if not os.path.exists(p):
            return None
        r = json.load(open(p))
        for task in ("los", "readmission"):
            k = HEADLINE[task][0]
            out.setdefault(task, []).append(r["tasks"][task]["dprepeat"][k])
    return out


def main():
    files = sorted(glob.glob(os.path.join(HERE, "results", "seeds", "seed_*.json")))
    if not files:
        raise SystemExit("no per-seed results yet")
    runs = [json.load(open(f)) for f in files]
    seeds = [r["seed"] for r in runs]
    print(f"aggregating {len(runs)} seeds: {seeds}\n")

    agg = {"seeds": seeds, "n_seeds": len(runs), "tasks": {}, "beddemand": {},
           "comparisons": {}}
    noise = _noise(seeds)
    agg["noise_control_available"] = noise is not None

    for task, metrics in HEADLINE.items():
        agg["tasks"][task] = {}
        series = {}
        for arm in ARMS:
            agg["tasks"][task][arm] = {}
            for met in metrics:
                vals = [r["tasks"][task]["arms"][arm][met] for r in runs]
                series[(arm, met)] = vals
                m, half, lo, hi = ci(vals)
                agg["tasks"][task][arm][met] = {
                    "mean": m, "ci95_half_width": half, "lo": lo, "hi": hi,
                    "values": vals}
        # MIA
        agg["tasks"][task]["mia"] = {}
        for arm in ("stage0", "stage1", "fairsample"):
            for att in ("attack_auc_dcr", "attack_auc_knn_classifier"):
                vals = [r["tasks"][task]["mia"][arm][att] for r in runs]
                m, half, lo, hi = ci(vals)
                agg["tasks"][task]["mia"].setdefault(arm, {})[att] = {
                    "mean": m, "ci95_half_width": half, "lo": lo, "hi": hi}

        # the decisive comparisons, do the programme's claims survive noise?
        key = metrics[0]
        cmps = {
            "cost_of_synthesis (TRTR vs stage0)": ("TRTR", "stage0"),
            "cost_of_privacy (stage0 vs stage1)": ("stage0", "stage1"),
            "cost_of_fair_sampling (stage1 vs fairsample)": ("stage1", "fairsample"),
            "cost_of_fair_training (stage1 vs fairtrain)": ("stage1", "fairtrain"),
        }
        agg["comparisons"][task] = {
            name: paired(series[(a, key)], series[(b, key)])
            for name, (a, b) in cmps.items()}
        # NEGATIVE CONTROL: same recipe, same split, independent model
        # randomness. This comparison SHOULD be null; if it isn't, no
        # between-arm claim in the programme is safe.
        if noise:
            agg["comparisons"][task]["NEGATIVE CONTROL (stage1 vs independent "
                                     "DP replicate)"] = paired(
                series[("stage1", key)], noise[task])
            agg["tasks"][task]["dprepeat"] = {key: dict(
                zip(("mean", "ci95_half_width", "lo", "hi"), ci(noise[task])))}
            agg["tasks"][task]["dprepeat"][key]["values"] = noise[task]
        agg["comparisons"][task]["_metric"] = key
        # NOTE: `dpbaseline` is bit-identical to `stage1` in this harness (same
        # class, same seed), kept as a determinism/reproducibility check, NOT
        # used as the noise control. See src/noise_control.py.

    for arm in ("TRTR", "stage0", "stage1"):
        agg["beddemand"][arm] = {}
        for met in ("mae", "rmse"):
            m, half, lo, hi = ci([r["beddemand"][arm][met] for r in runs])
            agg["beddemand"][arm][met] = {"mean": m, "ci95_half_width": half,
                                          "lo": lo, "hi": hi}

    with open(os.path.join(HERE, "results", "metrics_ci.json"), "w") as f:
        json.dump(agg, f, indent=2)
    _table(agg)
    print("wrote results/metrics_ci.json and results/results_table.md")


def _fmt(d):
    return f"{d['mean']:.3f} ± {d['ci95_half_width']:.3f}"


def _table(a):
    n = a["n_seeds"]
    L = [f"# Evidence hardening: {n} seeds, patient-level splits, mean ± 95% CI",
         "",
         f"Seeds: {a['seeds']}. Splits are patient-disjoint (the readmission "
         "cohort's row-level split leaked 42.7% of test rows). Every figure "
         "below replaces a single-seed number from Stages 0 to 2.", ""]

    L += ["## 30-day readmission: XGBoost", "",
          "| Arm | ROC-AUC | PR-AUC | Accuracy |", "|---|---|---|---|"]
    for arm in ARMS:
        r = a["tasks"]["readmission"][arm]
        L += [f"| {arm} | {_fmt(r['roc_auc'])} | {_fmt(r['pr_auc'])} "
              f"| {_fmt(r['accuracy'])} |"]

    L += ["", "## Length of stay: RandomForest", "",
          "| Arm | R² | RMSE (d) | MAE (d) |", "|---|---|---|---|"]
    for arm in ARMS:
        r = a["tasks"]["los"][arm]
        L += [f"| {arm} | {_fmt(r['r2'])} | {_fmt(r['rmse'])} | {_fmt(r['mae'])} |"]

    L += ["", "## Bed demand: Poisson-AR", "", "| Arm | MAE | RMSE |",
          "|---|---|---|"]
    for arm in ("TRTR", "stage0", "stage1"):
        r = a["beddemand"][arm]
        L += [f"| {arm} | {_fmt(r['mae'])} | {_fmt(r['rmse'])} |"]

    L += ["", "## Membership inference (0.5 = no leakage)", "",
          "| Task | Arm | DCR attack | kNN attack |", "|---|---|---|---|"]
    for task in ("los", "readmission"):
        for arm in ("stage0", "stage1", "fairsample"):
            m = a["tasks"][task]["mia"][arm]
            L += [f"| {task} | {arm} | {_fmt(m['attack_auc_dcr'])} "
                  f"| {_fmt(m['attack_auc_knn_classifier'])} |"]

    L += ["", "## Run-to-run instability: does DP make the generator unpredictable?",
          "", "| Task | Metric | sd(stage0, non-private) | sd(stage1, DP) | inflation |",
          "|---|---|---|---|---|"]
    for task in ("readmission", "los"):
        met = HEADLINE[task][0]
        s0 = np.std(a["tasks"][task]["stage0"][met]["values"], ddof=1)
        s1 = np.std(a["tasks"][task]["stage1"][met]["values"], ddof=1)
        L += [f"| {task} | {met} | {s0:.3f} | {s1:.3f} | **{s1/max(s0,1e-9):.1f}×** |"]

    L += ["", "## Do the claims survive seed noise? (paired t-test across seeds)",
          "", "| Task | Comparison | Δ | p | verdict |", "|---|---|---|---|---|"]
    for task in ("readmission", "los"):
        c = a["comparisons"][task]
        met = c["_metric"]
        for name, r in c.items():
            if name == "_metric" or r is None:
                continue
            v = "**SIGNIFICANT**" if r["significant"] else "within noise"
            L += [f"| {task} ({met}) | {name} | {r['mean_diff']:+.3f} "
                  f"| {r['p_value']:.4f} | {v} |"]
    if not a.get("noise_control_available"):
        L += ["", "> Negative control (independent DP replicate) pending, "
              "`make noise`. `dpbaseline` is bit-identical to `stage1` here "
              "(same class, same seed): a determinism check, not a noise "
              "control."]

    with open(os.path.join(HERE, "results", "results_table.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
