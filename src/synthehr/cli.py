"""
Command-line interface::

    synthehr demo     [--out DIR] [--quick]
    synthehr fit      --data train.csv --out model.synthehr [options]
    synthehr sample   --model model.synthehr --rows N --out synthetic.csv
    synthehr evaluate --train train.csv --test test.csv \\
                      --synthetic synthetic.csv --target COL --out DIR
    synthehr gate     --report DIR/report.json [--evidence DIR]

Run ``synthehr <command> --help`` for every option.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

import pandas as pd

from synthehr import __version__
from synthehr.compliance import ReleaseGate, build_evidence_pack
from synthehr.data import impute_with_train_medians, patient_disjoint_split
from synthehr.evaluation import Report, evaluate
from synthehr.generators import CTGAN, DPCTGAN, DPTabDDPM, TabDDPM, load

MODELS = {"tabddpm": TabDDPM, "dp-tabddpm": DPTabDDPM, "ctgan": CTGAN,
          "dp-ctgan": DPCTGAN}


def _list(s: Optional[str]) -> Optional[list]:
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def _build(args) -> object:
    cls = MODELS[args.model]
    kw = {"seed": args.seed, "verbose": not args.quiet}
    if args.model in ("tabddpm", "dp-tabddpm"):
        kw["device"] = args.device
        if args.size:
            kw["size"] = args.size
        if args.steps and args.model == "tabddpm":
            kw["steps"] = args.steps
    if args.model in ("ctgan", "dp-ctgan") and args.epochs:
        kw["epochs"] = args.epochs
    if args.model.startswith("dp-"):
        kw.update(epsilon=args.epsilon, delta=args.delta)
    if args.model == "dp-tabddpm":
        kw["privacy_unit"] = args.privacy_unit
        if args.epochs:
            kw["epochs"] = args.epochs
    return cls(**kw)


def cmd_fit(args) -> int:
    df = pd.read_csv(args.data)
    groups = None
    if args.group_column:
        groups = df.pop(args.group_column).to_numpy()
    if args.drop:
        df = df.drop(columns=_list(args.drop))
    disc = _list(args.discrete)
    df = impute_with_train_medians(df, discrete=disc or [])
    model = _build(args)
    t = time.time()
    model.fit(df, discrete_columns=disc, groups=groups)
    model.save(args.out)
    eps = getattr(model, "epsilon_", None)
    print(f"fitted {type(model).__name__} on {len(df):,} rows in "
          f"{time.time()-t:.0f}s -> {args.out}"
          + (f"  (epsilon {eps:.2f}, delta {model.target_delta:g})"
             if eps else "  (not private)"))
    return 0


def cmd_sample(args) -> int:
    model = load(args.model)
    df = model.sample(args.rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} synthetic rows -> {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    drop = _list(args.drop) or []
    frames = []
    for path in (args.train, args.test, args.synthetic):
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in drop if c in df.columns])
        df[args.target] = df[args.target].astype(str)
        frames.append(df)
    train, test, syn = frames
    gen = load(args.model) if args.model else None
    rep = evaluate(train, test, syn, target=args.target,
                   discrete_columns=_list(args.discrete),
                   sensitive=_list(args.sensitive) or (), generator=gen,
                   positive=args.positive, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    rep.to_json(os.path.join(args.out, "report.json"))
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(rep.to_markdown())
    print(rep.to_markdown())
    return 0


def cmd_gate(args) -> int:
    rep = Report.from_json(args.report)
    gate = ReleaseGate(max_epsilon=args.max_epsilon,
                       min_utility_ratio=args.min_utility_ratio)
    decision = gate.check(rep, signed_human_gates=_list(args.signed) or ())
    print(decision.to_markdown())
    if args.evidence:
        build_evidence_pack(rep, decision, args.evidence, name=args.name)
        print(f"evidence pack -> {args.evidence}")
    return 0 if decision.promoted else 2


def cmd_demo(args) -> int:
    """Fabricated cohort -> non-private and patient-level private TabDDPM ->
    evaluation -> gate -> evidence pack."""
    from synthehr.demo import DISCRETE, SENSITIVE, TARGET, make_demo_cohort
    q = args.quick
    out = args.out
    os.makedirs(out, exist_ok=True)
    cohort = make_demo_cohort(1500 if q else 6000, seed=args.seed)
    train, test = patient_disjoint_split(cohort, cohort["patient_id"],
                                         seed=args.seed)
    groups = train.pop("patient_id").to_numpy()
    test = test.drop(columns="patient_id")
    train.to_csv(os.path.join(out, "real_train.csv"), index=False)
    print(f"fabricated cohort: {len(train):,} training and {len(test):,} "
          f"held-out stays (patient-disjoint)")
    runs = {
        "tabddpm": TabDDPM(size="small" if q else "large",
                           steps=300 if q else 6000, timesteps=50 if q else 500,
                           device=args.device, seed=args.seed),
        "dp-tabddpm-patient": DPTabDDPM(
            epsilon=10.0, privacy_unit="patient", epochs=5 if q else 200,
            expected_batch=256 if q else 1024, timesteps=50 if q else 500,
            chunk=128 if q else 512, device=args.device, seed=args.seed),
    }
    gate = ReleaseGate(min_utility_ratio=0.25)
    for name, model in runs.items():
        t = time.time()
        model.fit(train, discrete_columns=DISCRETE, groups=groups)
        syn = model.sample(len(train))
        syn.to_csv(os.path.join(out, f"synthetic_{name}.csv"), index=False)
        rep = evaluate(train, test, syn, target=TARGET,
                       discrete_columns=DISCRETE, sensitive=SENSITIVE,
                       generator=model, n_candidates=1000,
                       min_group_size=30 if q else 100, seed=args.seed)
        decision = gate.check(rep)
        build_evidence_pack(rep, decision, os.path.join(out, name),
                            name=f"demo release ({name})")
        rep.to_json(os.path.join(out, name, "report.json"))
        print(f"\n=== {name} ({time.time()-t:.0f}s) ===")
        print(rep.to_markdown())
        print(f"gate: internal {decision.internal}, external "
              f"{decision.external}")
    print(f"\neverything written to {out}/")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="synthehr",
        description="Synthetic tabular health data with measured realism, "
                    "formal privacy and release evidence.")
    p.add_argument("--version", action="version",
                   version=f"synthehr {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end run on a fabricated cohort")
    d.add_argument("--out", default="synthehr-demo")
    d.add_argument("--quick", action="store_true",
                   help="tiny models (about a minute on a laptop CPU)")
    d.add_argument("--device", default="auto")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fit", help="train a generator on a CSV")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True, help="model file to write")
    f.add_argument("--model", choices=sorted(MODELS), default="tabddpm")
    f.add_argument("--discrete", help="comma-separated categorical columns")
    f.add_argument("--drop", help="comma-separated columns to ignore")
    f.add_argument("--group-column", help="patient id column (removed from "
                   "the data; required for --privacy-unit patient)")
    f.add_argument("--epsilon", type=float, default=10.0)
    f.add_argument("--delta", type=float, default=1e-5)
    f.add_argument("--privacy-unit", choices=["record", "patient"],
                   default="record")
    f.add_argument("--size", choices=["small", "medium", "large"])
    f.add_argument("--steps", type=int, help="TabDDPM optimiser steps")
    f.add_argument("--epochs", type=int)
    f.add_argument("--device", default="auto")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--quiet", action="store_true")
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("sample", help="draw rows from a fitted generator")
    s.add_argument("--model", required=True)
    s.add_argument("--rows", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser("evaluate", help="utility, realism and privacy report")
    e.add_argument("--train", required=True, help="the generator's training "
                   "data (attack members)")
    e.add_argument("--test", required=True, help="real rows the generator "
                   "never saw")
    e.add_argument("--synthetic", required=True)
    e.add_argument("--target", required=True)
    e.add_argument("--positive", default="1", help="positive class label")
    e.add_argument("--discrete")
    e.add_argument("--sensitive", help="comma-separated columns for fairness")
    e.add_argument("--drop", help="columns to ignore (for example ids)")
    e.add_argument("--model", help="the fitted model, to report its epsilon")
    e.add_argument("--out", default="report")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_evaluate)

    g = sub.add_parser("gate", help="release decision and evidence pack")
    g.add_argument("--report", required=True)
    g.add_argument("--max-epsilon", type=float, default=10.0)
    g.add_argument("--min-utility-ratio", type=float,
                   help="e.g. 0.5: synthetic utility at least half the real "
                        "ceiling")
    g.add_argument("--signed", help="human gates signed: data_use,governance")
    g.add_argument("--evidence", help="directory for the evidence pack")
    g.add_argument("--name", default="synthetic release")
    g.set_defaults(func=cmd_gate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
