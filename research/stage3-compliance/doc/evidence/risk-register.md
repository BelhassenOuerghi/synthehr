# Residual risk register (Stage 3)

*Compiled 2026-07-12. These are the risks that remain after the mitigations applied
in Stages 0 to 2, each with a severity, a current status and a planned treatment.
Disclosing them openly is itself one of the release gate's criteria.*

| Ref | Residual risk | Severity | Status | Treatment |
|---|---|---|---|---|
| RR-1 | **Row-level rather than patient-level adjacency.** The privacy budget protects one ICU stay. A patient with several stays therefore receives weaker protection than the per-stay bound implies. | Medium | Open | Group privacy, or per-patient sampling. Owner: candidate. |
| RR-2 | **Non-private preprocessing statistics.** The variational Gaussian-mixture normalisation parameters and the discrete category schema are fitted on real data, outside the privacy boundary. | Medium | Open | Differentially private quantile normalisation, and a private schema release. Owner: candidate. |
| RR-3 | **An approximation in the accounting.** Fixed-size batches drawn without replacement are accounted as Poisson subsampling. | Low | Open | A tight amplification-without-replacement bound, or true Poisson batches. |
| RR-4 | **Empirical attacks are not proofs.** Attacks at chance (0.499) show that weak black-box adversaries fail. Stronger adversaries are untested, and attacks cannot certify a release; only the formal bound can. | Medium | Open | Shadow-model, white-box and attribute-inference attacks. |
| RR-5 | **Run-to-run instability under differential privacy.** The generator's utility and its subgroup representation vary substantially between training runs at the same budget, with the standard deviation inflated by up to an order of magnitude. A single run cannot tell a controller what they are about to release. | High | Open | Report all figures with confidence intervals over multiple seeds, which is now programme policy. Investigate ensembling across independent runs. |
| RR-6 | **Representation is rebalanced, not restored.** The released Medicare share is 0.368 against 0.542 in the real cohort, and it varies between runs. | Medium | Open | The realism pillar, for better conditional obedience, and intersectional balancing. |
| RR-7 | **Low utility invites misuse.** A released readmission AUC of 0.451 could be mistaken for a deployable model. | High | Mitigated | Prohibitions in the model and data cards, an explicit ban on clinical use, and honest disclosure of utility. |
| RR-8 | **Contractual limits on redistribution.** External sharing of MIMIC-derived synthetic data may be restricted by the data use agreement. | High | Open, and blocking | The `dua` human gate. External release is held pending legal review. |
