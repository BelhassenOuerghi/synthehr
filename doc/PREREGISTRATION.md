# Pillar 4 pre-registration

*Written before the main experiment was run, and committed before its first
result existed. The only data seen at this point is the pilot in
[`results/pilot.json`](../results/pilot.json), which used seed 42's training rows
and a validation split carved from them. No test set of any seed had been
touched.*

## Why pre-register

This programme has retracted two of its own headline claims, both because a
single favourable run was reported as if it were the distribution. Pillar 4 is
the pillar most exposed to that failure, because it is the one that is supposed
to deliver good news. So the questions, the arms, the tests and the conditions
under which each claim is dropped are fixed here first.

## Questions and hypotheses

**RQ1. Realism without privacy.** Does a diffusion generator (TabDDPM) recover
the signal that CTGAN synthesis destroys?
*H1:* non-private TabDDPM beats non-private CTGAN on downstream utility
(readmission ROC-AUC, length-of-stay R²) and on realism (detection AUC lower;
precision and recall higher).
*Dropped if:* the paired difference is not significant on either task.

**RQ2. Realism under privacy.** At the release budget (ε = 10, with exactly
Stage 1's noise convention), does the better generator survive privacy?
*H2:* DP-TabDDPM beats DP-CTGAN on utility and realism, and is more predictable
(smaller run-to-run spread).
*Dropped if:* not significant. A null here is itself a finding: a better
generator would not survive privacy.

**RQ3. Ensembling across DP runs.** Does pooling K = 4 independently trained DP
generators buy back predictability?
* *Full budget:* four runs at ε = 10 each, pooled (composed ε about 23). Does
  pooling shrink the spread across seeds, and what does it do to the mean?
* *Split budget:* four runs whose composition is exactly ε = 10. At a fixed
  privacy cost, is one generator or a pool of four noisier ones the better
  release?

Both regimes are measured for both generator families.
*H3a:* pooling at full budget reduces the spread.
*H3b:* at split budget, pooling does not beat a single run on mean utility. This
is a prediction of a null, stated as such.

**RQ4. Do realism and privacy commute?** The capacity that maximises realism
without privacy need not be the one that maximises it under privacy, because
DP-SGD noise grows with the parameter count. The arm is denoiser capacity
S/M/L × {non-private, DP}.
*H4:* a capacity × privacy interaction exists. The difference-in-differences
`(L − S)_nonprivate − (L − S)_DP` is positive (a large model helps without
privacy and hurts, or helps less, under it).
*Dropped if:* the interaction is not significant. The ordering of the ladder
then has to be defended by argument alone, as before.

**RQ5. Is fairness measurable once utility exists?** Stage 2's fairness
metrics were vacuous because the classifiers predicted no positives at all.
With a release whose utility is restored, fair resampling (zero extra ε) is
applied to both DP generators.
*H5:* on DP-TabDDPM the downstream classifier is non-degenerate (positive
prediction rate between 2% and 98% in every subgroup), so subgroup AUC gaps,
demographic parity and equalised odds become measurable. The cost of fair
resampling is reported against the noise floor, as Stage 2's was.

**RQ6. Patient-level privacy.** Stages 1-2 protected an ICU stay; the ethical
subject is a patient, who can contribute up to 41 stays. What does moving the
guarantee to the patient (risk register RR-1) cost at ε = 10?
*H6:* the cost is within the noise floor. Grouping changes the sampling unit
for the 28-30% of fitting rows that come from patients with more than one stay
in the fitting sample (up to 26 stays for one patient; measured on all five
seeds), and per-patient clipping bounds those patients' combined influence.

## Arms (per task, per seed)

| Family | Arm | What it is |
|---|---|---|
| reference | `TRTR` | the real training data itself: utility ceiling, real-vs-real realism control, and positive control for the attacks |
| CTGAN | `ctgan` | Stage 0 recipe (non-private) |
| | `dpctgan` | Stage 1 recipe at ε = 10; reproduces the evidence repository's `stage1` |
| | `dpctgan_r1..r3` | independent replicates (same fitting rows, new model seed) |
| | `dpctgan_s1..s4` | four runs at the split budget |
| | `dpctgan_ens4`, `dpctgan_ens4split` | pooled releases (full and split budget) |
| | `dpctgan_rake` | fair resampling of `dpctgan` (zero extra ε) |
| TabDDPM | `ddpm`, `ddpm_S`, `ddpm_L` | non-private, capacities M, S, L |
| | `dpddpm`, `dpddpm_S`, `dpddpm_L` | DP at ε = 10, capacities M, S, L |
| | `dpddpm_r1..r3`, `dpddpm_s1..s4`, ensembles, `dpddpm_rake` | as for CTGAN |
| | `dpddpm_patient` | DP at ε = 10 with the patient as the privacy unit |

Seeds 42 to 46, the same as the evidence repository, so every split is the
same patient-disjoint split and every comparison with Stages 0-2 is paired.
Every generator is fitted on the frozen harness's 30,000-row fitting sample
and releases as many rows as the real training set.

## Metrics

* **Utility:** the frozen downstream models (XGBoost for readmission, random
  forest for length of stay), trained on each arm and tested on the untouched
  real test split.
* **Realism:** detection AUC (classifier two-sample test), precision and
  recall, marginal distances, and correlation and feature-target errors, all
  against real rows the generator never saw. `TRTR` gives the real-vs-real
  ceiling.
* **Privacy:** the frozen DCR and k-NN attacks with patient-disjoint members
  and non-members, plus the true-positive rate at 1% and 0.1% false positives.
  ε is reported in Stage 1's convention, and also under add/remove adjacency.
* **Fairness:** the frozen subgroup metrics, with a degeneracy check.

## Decision rules

1. **Significance.** A between-arm difference is reported as a finding only if
   a paired t-test across the five seeds gives p < 0.05. Holm-adjusted p-values
   are reported beside the raw ones for the family of RQ1 to RQ6 primary
   comparisons.
2. **Noise floor.** It must also exceed the arm's own noise floor: the mean
   absolute difference between independent replicates of the same recipe on
   the same split (`dp*` against `dp*_r1..r3`). A significant effect smaller
   than the floor is reported, but flagged as practically unresolved.
3. **Predictability.** Spread is compared with the Brown-Forsythe test. Single
   runs contribute four replicates per seed (20 runs), and pooled releases one
   per seed (5). With n = 5 the test is weak, and a null is reported as
   "not resolved", not as "no effect".
4. **Everything is reported.** Every arm, every seed and every null. Per-seed
   values are committed.

## Hyperparameters

Fixed by the pilot and frozen in [`config.yaml`](../config.yaml) before the main
run. See [the pilot section of the paper](research-paper.md#pilot) for the
table. Choosing DP hyperparameters on private data is itself a small leak that
ε does not cover (Papernot & Steinke, 2022). It is disclosed here rather than
hidden.
