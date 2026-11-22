"""
Evidence as code: compile the release evidence from the measurements.

Compliance documents written by hand drift: the pipeline is re-run and the
document silently stops describing the data. Here every number is injected
from the :class:`~synthehr.evaluation.Report` and the gate's
:class:`~synthehr.compliance.gate.Decision`, and rendering fails if any
placeholder is left unfilled.

Writes ``evidence.json`` (everything, machine-readable), ``data-card.md``
(what the release is and how far to trust it) and ``release-decision.md``.
These are evidence for a human reviewer, not legal advice.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from string import Template
from typing import Optional

from synthehr._version import __version__ as _VERSION
from synthehr.compliance.gate import Decision
from synthehr.evaluation.report import Report

_DATA_CARD = Template("""\
# Data card: $name

Generated $date by synthehr $version.

## What this is

A synthetic table of $n_synthetic rows produced by `$generator` from a real
training table of $n_train rows. It is intended for $intended_use. It must not
be used for clinical decisions about individuals.

## Privacy

Formal guarantee: $privacy_formal. Privacy unit: $privacy_unit.
Membership-inference attacks (members: training rows; non-members: $n_test
held-out rows): distance-to-closest-record AUC $mia_dcr, k-NN classifier AUC
$mia_knn (0.5 is chance), true-positive rate at 1% false positives $mia_tpr.
Share of synthetic rows within 0.1 (standardised) of a training row:
$near_dup.

Attacks at chance are not proof of privacy: they also sit at chance for
non-private generators. Only the formal guarantee bounds what any attacker
can learn.

## Fidelity

Detection AUC $det (real-versus-real control $det_ctl; 0.5 means
indistinguishable). Coverage $cov (control $cov_ctl). Precision $prec.
Mean per-column Wasserstein distance $w1 standard deviations; mean
categorical total-variation distance $tvd.

## Utility

A model trained on this release and tested on real held-out data reaches
$metric $util, against $util_ceiling when trained on the real data
($ratio of the real ceiling).

## Fairness

$fairness

## Limitations

Preprocessing fitted on the real data (category lists, numeric transforms) is
not covered by the formal guarantee. Results apply to the data and settings
above; a new dataset needs a new evaluation.
""")


def _fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def build_evidence_pack(report: Report, decision: Decision, out_dir: str,
                        name: str = "synthetic release",
                        intended_use: str = "method development, teaching "
                                            "and software testing",
                        date: Optional[str] = None) -> dict:
    """Write the evidence pack to ``out_dir`` and return its values."""
    os.makedirs(out_dir, exist_ok=True)
    p, r, rc = report.privacy, report.realism, report.realism_control or {}
    k = report.headline_metric
    ratio = report.utility_ratio()
    fair_lines = []
    for col, f in report.fairness.items():
        gaps = ", ".join(f"{g} {_fmt(f[g])}" for g in
                         ("auc_gap", "dp_diff", "eo_diff", "rmse_gap",
                          "bias_gap") if g in f)
        warn = (" The downstream classifier is degenerate (it predicts one "
                "class in some group), so parity metrics are vacuous."
                if f.get("degenerate") else "")
        gaps = gaps or ("not computed: fewer than two groups reach the "
                        "minimum group size in the held-out data")
        fair_lines.append(f"By `{col}`: {gaps}.{warn}")
    values = {
        "name": name, "date": date or _dt.date.today().isoformat(),
        "version": _VERSION, "generator": report.generator or "unknown",
        "n_synthetic": f"{report.n_synthetic:,}",
        "n_train": f"{report.n_train:,}", "n_test": f"{report.n_test:,}",
        "intended_use": intended_use,
        "privacy_formal": ("none: this release is not differentially private"
                           if p.get("epsilon") is None else
                           f"({p['epsilon']:.2f}, {p['delta']:g})-differential "
                           "privacy"),
        "privacy_unit": p.get("privacy_unit", "not applicable"),
        "mia_dcr": _fmt(p["attack_auc_dcr"]),
        "mia_knn": _fmt(p["attack_auc_knn_classifier"]),
        "mia_tpr": _fmt(p["tpr@fpr=0.01"]),
        "near_dup": _fmt(p["near_duplicate_share@0.1"]),
        "det": _fmt(r.get("detection_auc")),
        "det_ctl": _fmt(rc.get("detection_auc")),
        "cov": _fmt(r.get("coverage")), "cov_ctl": _fmt(rc.get("coverage")),
        "prec": _fmt(r.get("precision")), "w1": _fmt(r.get("marginal_w1")),
        "tvd": _fmt(r.get("marginal_tvd")), "metric": k,
        "util": _fmt(report.utility[k]),
        "util_ceiling": _fmt((report.utility_ceiling or {}).get(k)),
        "ratio": "n/a" if ratio is None else f"{100 * ratio:.0f}%",
        "fairness": ("\n\n".join(fair_lines) if fair_lines else
                     "No sensitive columns were evaluated."),
    }
    card = _DATA_CARD.substitute(values)          # raises on a missing value
    with open(os.path.join(out_dir, "data-card.md"), "w") as f:
        f.write(card)
    with open(os.path.join(out_dir, "release-decision.md"), "w") as f:
        f.write(decision.to_markdown())
    with open(os.path.join(out_dir, "evidence.json"), "w") as f:
        json.dump({"report": report.to_dict(), "decision": decision.to_dict(),
                   "values": values}, f, indent=2, default=float)
    return values
