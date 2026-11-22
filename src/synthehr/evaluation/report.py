"""
One call to measure a synthetic release the way the research did.

``evaluate(real_train, real_test, synthetic, target=...)`` returns a
:class:`Report` with:

* **utility**: train-on-synthetic, test-on-real, with the train-on-real
  ceiling beside it;
* **realism**: detection AUC, precision/recall, density/coverage, marginal
  and dependence errors, label rate, with the real-versus-real control;
* **privacy**: membership-inference attacks (training rows as members, the
  held-out rows as non-members) and, if the generator is passed, its formal
  epsilon and delta;
* **fairness**: subgroup metrics and representation for ``sensitive``
  columns, with a degeneracy flag.

``real_train`` must be the data the generator was trained on (it supplies
the attack's members) and ``real_test`` real rows it never saw, ideally
patient-disjoint. Missing values in every frame are filled with the training
medians.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import pandas as pd

from synthehr.data import impute_with_train_medians
from synthehr.evaluation.realism import label_rate, realism
from synthehr.evaluation.utility import infer_task, tstr
from synthehr.fairness.metrics import representation
from synthehr.generators._base import infer_discrete_columns
from synthehr.privacy.attacks import membership_inference


@dataclass
class Report:
    target: str
    task: str
    n_train: int
    n_test: int
    n_synthetic: int
    utility: dict
    utility_ceiling: Optional[dict]
    realism: dict
    realism_control: Optional[dict]
    privacy: dict
    fairness: dict = field(default_factory=dict)
    representation: dict = field(default_factory=dict)
    label_rate: Optional[dict] = None
    generator: Optional[str] = None

    # -- the one number per axis that a reader looks for first ---------------
    @property
    def headline_metric(self) -> str:
        return "roc_auc" if self.task == "classification" else "r2"

    def utility_ratio(self) -> Optional[float]:
        """Synthetic utility as a share of the real ceiling (above chance for
        classification, above zero for R²)."""
        if not self.utility_ceiling:
            return None
        k = self.headline_metric
        base = 0.5 if k == "roc_auc" else 0.0
        top = self.utility_ceiling[k] - base
        return None if top <= 0 else (self.utility[k] - base) / top

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "Report":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2, default=float)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    def to_markdown(self) -> str:
        k = self.headline_metric
        r, rc = self.realism, self.realism_control or {}
        p = self.privacy
        eps = p.get("epsilon")
        eps_txt = ("none (not private)" if eps is None
                   else f"{eps:.2f} (delta {p.get('delta')})")
        lines = [
            f"# Synthetic data report: {self.generator or 'release'}", "",
            f"Target `{self.target}` ({self.task}); {self.n_train:,} training, "
            f"{self.n_test:,} held-out and {self.n_synthetic:,} synthetic rows.",
            "", "| Axis | Metric | Synthetic | Real data (ceiling or control) |",
            "|---|---|---|---|",
            f"| Utility | {k} (train on synthetic, test on real) | "
            f"{self.utility[k]:.3f} | "
            f"{(self.utility_ceiling or {}).get(k, float('nan')):.3f} |",
            f"| Realism | detection AUC (0.5 = indistinguishable) | "
            f"{r['detection_auc']:.3f} | {rc.get('detection_auc', float('nan')):.3f} |",
            f"| Realism | coverage | {r['coverage']:.3f} | "
            f"{rc.get('coverage', float('nan')):.3f} |",
            f"| Realism | precision | {r['precision']:.3f} | "
            f"{rc.get('precision', float('nan')):.3f} |",
            f"| Privacy | DCR attack AUC (0.5 = chance) | "
            f"{p['attack_auc_dcr']:.3f} | – |",
            f"| Privacy | attack TPR at 1% FPR (0.01 = chance) | "
            f"{p['tpr@fpr=0.01']:.3f} | – |",
            f"| Privacy | formal epsilon | {eps_txt} | – |",
        ]
        if self.label_rate:
            lines.append(f"| Realism | label rate | "
                         f"{self.label_rate['synthetic']:.3f} | "
                         f"{self.label_rate['real']:.3f} |")
        ratio = self.utility_ratio()
        if ratio is not None:
            lines += ["", f"Utility recovered: {100 * ratio:.0f}% of the real "
                      "ceiling."]
        for c, fm in self.fairness.items():
            flag = (" (degenerate classifier: parity metrics are vacuous)"
                    if fm.get("degenerate") else "")
            gaps = ", ".join(f"{g} {fm[g]:.3f}" for g in
                             ("auc_gap", "dp_diff", "eo_diff", "rmse_gap",
                              "bias_gap") if g in fm)
            if not gaps:
                gaps = ("not computed: fewer than two groups reach the "
                        "minimum group size in the held-out data")
            lines.append(f"\nFairness by `{c}`: {gaps}{flag}.")
        return "\n".join(lines) + "\n"


def evaluate(real_train: pd.DataFrame, real_test: pd.DataFrame,
             synthetic: pd.DataFrame, target: str,
             discrete_columns: Optional[Sequence[str]] = None,
             task: str = "auto", positive="1",
             sensitive: Sequence[str] = (), generator=None,
             control: bool = True, n_candidates: int = 8000,
             min_group_size: int = 200, seed: int = 0) -> Report:
    """Measure ``synthetic`` against real data. See the module docstring."""
    disc = (list(discrete_columns) if discrete_columns is not None
            else infer_discrete_columns(real_train))
    cols = list(real_train.columns)
    real_train, real_test, synthetic = impute_with_train_medians(
        real_train[cols], real_test[cols], synthetic[cols], discrete=disc)
    task = infer_task(real_test[target]) if task == "auto" else task

    util = tstr(synthetic, real_test, target, disc, task, positive,
                sensitive, seed, min_group_size=min_group_size)
    fairness = util.pop("fairness", {})
    ceiling = (tstr(real_train, real_test, target, disc, task, positive, (),
                    seed) if control else None)
    rl = realism(real_train, real_test, synthetic, disc, target, seed)
    ctl = (realism(real_train, real_test, real_train, disc, target, seed)
           if control else None)
    priv = membership_inference(real_train, real_test, synthetic,
                                [c for c in disc if c != target],
                                n_candidates=n_candidates, seed=seed)
    if generator is not None and getattr(generator, "epsilon_", None):
        priv["epsilon"] = float(generator.epsilon_)
        priv["delta"] = float(generator.target_delta)
        if getattr(generator, "epsilon_add_remove_", None):
            priv["epsilon_add_remove"] = float(generator.epsilon_add_remove_)
        priv["privacy_unit"] = getattr(generator, "unit", "record")
    else:
        priv["epsilon"] = None
    lr = None
    if task == "classification":
        lr = {"synthetic": label_rate(synthetic, target, positive),
              "real": label_rate(real_train, target, positive)}
    return Report(
        target=target, task=task, n_train=len(real_train),
        n_test=len(real_test), n_synthetic=len(synthetic), utility=util,
        utility_ceiling=ceiling, realism=rl, realism_control=ctl,
        privacy=priv, fairness=fairness,
        representation=representation(synthetic, real_train, sensitive),
        label_rate=lr,
        generator=type(generator).__name__ if generator is not None else None)
