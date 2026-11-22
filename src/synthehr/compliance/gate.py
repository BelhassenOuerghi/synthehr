"""
A release gate that can say no.

Technical checks decide whether a synthetic release may be promoted inside
the environment that holds the real data. External release additionally
needs human sign-off (data-use agreement and governance review); code never
closes those gates, so the external decision is HOLD until a person records
both in ``signed_human_gates``.

Checks:

* **formal privacy:** epsilon and delta within the ceilings. A release with no
  formal guarantee fails, whatever the attacks say: the research found that
  membership attacks sit at chance even against non-private generators, so
  an empirical pass cannot certify a release.
* **empirical privacy:** both attack AUCs inside a chance band, the
  true-positive rate at 1% false positives near chance, and few near-copies.
* **fairness evidence:** subgroup metrics were computed, and the classifier is
  not degenerate (a model that predicts one class is trivially "fair").
* **fitness for purpose** (optional, recommended): synthetic utility is at
  least a given share of the real ceiling. The research showed why: without
  it, the gate promoted every private release, including ones with no
  utility, and refused every realistic one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from synthehr.evaluation.report import Report

HUMAN_GATES = (
    ("data_use", "Data-use agreement or contractual review of redistributing "
                 "data derived from the source"),
    ("governance", "Practitioner or governance review of this evidence"),
)


@dataclass
class Check:
    id: str
    passed: bool
    evidence: str


@dataclass
class Decision:
    internal: str                      # PROMOTE or REFUSE
    external: str                      # HOLD or RELEASE
    checks: list = field(default_factory=list)
    open_human_gates: list = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.internal == "PROMOTE"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        rows = [f"| {c.id} | {'pass' if c.passed else 'FAIL'} | {c.evidence} |"
                for c in self.checks]
        gates = "".join(f"\n* {g}" for g in self.open_human_gates) or " none"
        return ("## Release decision\n\n"
                f"Internal promotion: **{self.internal}**. External release: "
                f"**{self.external}**.\n\n| Check | Result | Evidence |\n"
                "|---|---|---|\n" + "\n".join(rows) +
                f"\n\nHuman gates still open:{gates}\n")


@dataclass
class ReleaseGate:
    max_epsilon: float = 10.0
    max_delta: float = 1e-5
    attack_band: tuple = (0.45, 0.55)
    max_tpr_at_1pct: float = 0.03
    max_near_duplicate: float = 0.01
    min_utility_ratio: Optional[float] = None
    require_fairness_evidence: bool = True

    def check(self, report: Report,
              signed_human_gates: Sequence[str] = ()) -> Decision:
        p = report.privacy
        checks = []
        eps, delta = p.get("epsilon"), p.get("delta")
        checks.append(Check(
            "formal_privacy",
            eps is not None and eps <= self.max_epsilon + 1e-9
            and delta is not None and delta <= self.max_delta,
            "no formal guarantee" if eps is None else
            f"epsilon {eps:.2f} <= {self.max_epsilon}, delta {delta:g} <= "
            f"{self.max_delta:g}"))
        lo, hi = self.attack_band
        aucs = (p["attack_auc_dcr"], p["attack_auc_knn_classifier"])
        checks.append(Check(
            "empirical_privacy",
            all(lo <= a <= hi for a in aucs)
            and p["tpr@fpr=0.01"] <= self.max_tpr_at_1pct
            and p["near_duplicate_share@0.1"] <= self.max_near_duplicate,
            f"attack AUCs {aucs[0]:.3f}/{aucs[1]:.3f} in [{lo}, {hi}]; TPR at "
            f"1% FPR {p['tpr@fpr=0.01']:.3f}; near-copies "
            f"{p['near_duplicate_share@0.1']:.3f}"))
        if self.require_fairness_evidence:
            degenerate = [c for c, f in report.fairness.items()
                          if f.get("degenerate")]
            checks.append(Check(
                "fairness_evidence",
                bool(report.fairness) and not degenerate,
                "no sensitive columns evaluated" if not report.fairness else
                (f"degenerate classifier for {degenerate}: parity metrics "
                 "are vacuous" if degenerate else
                 f"subgroup metrics for {list(report.fairness)}")))
        if self.min_utility_ratio is not None:
            ratio = report.utility_ratio()
            checks.append(Check(
                "fitness_for_purpose",
                ratio is not None and ratio >= self.min_utility_ratio,
                "no real ceiling to compare with" if ratio is None else
                f"{100 * ratio:.0f}% of the real ceiling (minimum "
                f"{100 * self.min_utility_ratio:.0f}%)"))
        promoted = all(c.passed for c in checks)
        open_gates = [name for gid, name in HUMAN_GATES
                      if gid not in signed_human_gates]
        return Decision(
            internal="PROMOTE" if promoted else "REFUSE",
            external="RELEASE" if promoted and not open_gates else "HOLD",
            checks=checks, open_human_gates=open_gates)
