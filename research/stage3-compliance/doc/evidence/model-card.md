# Model Card: DP-Fair-CTGAN generator (Stage 3)

*Structure follows Model Cards for Model Reporting (Mitchell et al., 2019).
Compiled 2026-07-12, with figures injected from the evaluation metrics.*

## Model details

Fair-DP-CTGAN (fair conditional generation on a DP-SGD critic; RDP-accounted). A from-scratch, MIT-licensed Conditional Tabular GAN whose WGAN-GP
critic is trained under DP-SGD, with fair conditional sampling layered on top. Bed
demand uses a differentially private Poisson-AR counterpart. Privacy budgets:
9.99 for readmission and 9.99 for length of stay, both at a delta of
1e-5, with a noise multiplier of 1.01. Author: Belhassen Ouerghi.

## Intended use

* **Intended.** To generate synthetic EHR for methodological research into the
  trade-off between privacy, fairness and utility, and to produce training data that
  is shareable in principle for the three ICU prediction tasks.
* **Out of scope.** Any clinical decision support, any deployment on real patients,
  and any use of the synthetic data as though it were real. This is not a medical
  device.

## Factors

Evaluated across the sensitive subgroups gender; insurance (readmission), gender (LOS), on three downstream tasks:
length of stay (regression), 30-day readmission (classification) and bed demand
(forecasting).

## Metrics

Trained on synthetic data and tested on real data, for the released
`fairsample` arm.

| Task | Real-data ceiling | Non-private synthesis | This release |
|---|---|---|---|
| Readmission ROC-AUC | 0.770 | 0.690 | 0.451 |
| Length of stay R² | 0.186 | 0.054 | -0.192 |

These are the measured values of the released dataset, which is a single training run
(seed 42). Differential privacy makes runs unpredictable, so a single run is not the
recipe's performance. Across 5 seeds on the same patient-disjoint splits (mean ±
95% CI, from the evidence repository) the same recipe gives:

| Task | Real-data ceiling | Non-private synthesis | This release's recipe |
|---|---|---|---|
| Readmission ROC-AUC | 0.778 ± 0.007 | 0.690 ± 0.018 | 0.484 ± 0.039 |
| Length of stay R² | 0.186 ± 0.001 | 0.009 ± 0.083 | -0.728 ± 1.701 |

Across seeds the non-private length-of-stay R² ranges from -0.104 to
0.054, so the seed-42 value above should not be read as typical. A
difference between two runs smaller than the measured noise floor (0.151
ROC-AUC, 3.080 R², minimum detectable paired effect at 5 seeds)
is not a finding.

Privacy metrics: membership-inference attack AUC of 0.497 and
0.503 for length of stay, and 0.499 and 0.501 for
readmission, under the two attacks; near-duplicate share of 0.000 and
0.000 respectively.

## Quantitative analyses: fairness

Fair conditional sampling rebalances subgroup representation toward a uniform target
at no additional privacy cost, and it reduces the run-to-run variability of that
representation. It does not restore the population distribution, and it should not be
described as repairing a systematic bias: under differential privacy, subgroup shares
vary widely between training runs rather than collapsing in a fixed direction.
Medicare share: 0.421 in the unmitigated arm and
0.368 in this release, against 0.542 in the real
cohort. Across 5 seeds: 0.406 ± 0.277 unmitigated and
0.350 ± 0.184 with fair sampling.

At this budget the downstream classifier's utility is low enough that group-parity
metrics are partly vacuous, since a model that predicts no positives is trivially
fair. Disparity is therefore reported jointly with utility and never on its own.

## Ethical considerations and caveats

Negative results are disclosed rather than suppressed. At this budget the released
readmission model performs far below the real-data ceiling and must not be mistaken
for a usable clinical model. Results are subject to substantial run-to-run variance,
which differential privacy inflates by up to an order of magnitude, so single-run
figures should not be relied upon. Residual non-private preprocessing statistics are
documented in the risk register. Clinical use is prohibited.
