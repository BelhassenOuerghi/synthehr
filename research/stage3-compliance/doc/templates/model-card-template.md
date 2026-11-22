# Model Card: DP-Fair-CTGAN generator (Stage 3)

*Structure follows Model Cards for Model Reporting (Mitchell et al., 2019).
Compiled {{GEN_DATE}}, with figures injected from the evaluation metrics.*

## Model details

{{GENERATOR}}. A from-scratch, MIT-licensed Conditional Tabular GAN whose WGAN-GP
critic is trained under DP-SGD, with fair conditional sampling layered on top. Bed
demand uses a differentially private Poisson-AR counterpart. Privacy budgets:
{{EPS_READM}} for readmission and {{EPS_LOS}} for length of stay, both at a delta of
{{DELTA}}, with a noise multiplier of {{SIGMA_TAB}}. Author: Belhassen Ouerghi.

## Intended use

* **Intended.** To generate synthetic EHR for methodological research into the
  trade-off between privacy, fairness and utility, and to produce training data that
  is shareable in principle for the three ICU prediction tasks.
* **Out of scope.** Any clinical decision support, any deployment on real patients,
  and any use of the synthetic data as though it were real. This is not a medical
  device.

## Factors

Evaluated across the sensitive subgroups {{SENSITIVE}}, on three downstream tasks:
length of stay (regression), 30-day readmission (classification) and bed demand
(forecasting).

## Metrics

Trained on synthetic data and tested on real data, for the released
`{{CANONICAL_ARM}}` arm.

| Task | Real-data ceiling | Non-private synthesis | This release |
|---|---|---|---|
| Readmission ROC-AUC | {{UTIL_TRTR_RD}} | {{UTIL_S0_RD}} | {{UTIL_REL_RD}} |
| Length of stay R² | {{UTIL_TRTR_LOS}} | {{UTIL_S0_LOS}} | {{UTIL_REL_LOS}} |

These are the measured values of the released dataset, which is a single training run
(seed 42). Differential privacy makes runs unpredictable, so a single run is not the
recipe's performance. Across {{MS_N}} seeds on the same patient-disjoint splits (mean ±
95% CI, from the evidence repository) the same recipe gives:

| Task | Real-data ceiling | Non-private synthesis | This release's recipe |
|---|---|---|---|
| Readmission ROC-AUC | {{MS_TRTR_RD}} | {{MS_S0_RD}} | {{MS_REL_RD}} |
| Length of stay R² | {{MS_TRTR_LOS}} | {{MS_S0_LOS}} | {{MS_REL_LOS}} |

Across seeds the non-private length-of-stay R² ranges from {{MS_S0_LOS_MIN}} to
{{MS_S0_LOS_MAX}}, so the seed-42 value above should not be read as typical. A
difference between two runs smaller than the measured noise floor ({{NOISE_FLOOR_RD}}
ROC-AUC, {{NOISE_FLOOR_LOS}} R², minimum detectable paired effect at {{MS_N}} seeds)
is not a finding.

Privacy metrics: membership-inference attack AUC of {{MIA_LOS_DCR}} and
{{MIA_LOS_KNN}} for length of stay, and {{MIA_RD_DCR}} and {{MIA_RD_KNN}} for
readmission, under the two attacks; near-duplicate share of {{NEARDUP_LOS}} and
{{NEARDUP_RD}} respectively.

## Quantitative analyses: fairness

Fair conditional sampling rebalances subgroup representation toward a uniform target
at no additional privacy cost, and it reduces the run-to-run variability of that
representation. It does not restore the population distribution, and it should not be
described as repairing a systematic bias: under differential privacy, subgroup shares
vary widely between training runs rather than collapsing in a fixed direction.
Medicare share: {{REP_MEDICARE_BASE}} in the unmitigated arm and
{{REP_MEDICARE_REL}} in this release, against {{REP_MEDICARE_REAL}} in the real
cohort. Across {{MS_N}} seeds: {{MS_MEDICARE_BASE}} unmitigated and
{{MS_MEDICARE_REL}} with fair sampling.

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
