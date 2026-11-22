# Differentially-Private Synthetic Electronic Health Records: the Privacy Pillar of a Four-Pillar Programme

Belhassen Ouerghi. Professional Engineering Doctorate, University of Hertfordshire.

Working paper. Stage 1 (privacy pillar) of the thesis *Advanced Data Generation for
Healthcare Predictive Modelling: Ensuring Privacy, High Realism, Bias Mitigation
and Ethical Compliance.*

---

## Abstract

Stage 0 of this programme established the utility baseline for synthetic EHR: a
from-scratch CTGAN fitted to the cohorts behind three MIMIC-IV prediction tasks and
scored by train-on-synthetic, test-on-real (TSTR) against a train-on-real (TRTR)
ceiling. That baseline carries no privacy guarantee, and its outputs consequently
cannot be released. This paper adds the guarantee. I retrain the identical
generator under differential privacy, using DP-SGD on the WGAN-GP critic with a
self-implemented Rényi accountant for the sampled Gaussian mechanism, at a grid of
budgets, and evaluate each resulting dataset along two axes: the Stage-0 utility
protocol unchanged, and membership-inference attacks with real training rows as
members and held-out rows as non-members. The deliverable is the programme's first
privacy-utility frontier. Two findings are reported that a single training run
cannot produce. First, black-box membership-inference attacks sit at chance at
every budget, including the unbounded one, so such attacks cannot certify a
release and the formal bound must carry the argument. Second, differential privacy
at practical budgets does not merely cost utility on average, it destroys the
predictability of the generator, inflating run-to-run variance by up to an order of
magnitude, with the consequence that reporting a single DP run as the cost of
privacy is unsound.

---

## 1. Motivation: from utility to releasability

Stage 0 answered the question of whether the synthetic cohort still trains a useful
model. This stage answers the question a data controller actually has to put to a
regulator: can this synthetic dataset leave the building? Those are different
questions. A GAN with no privacy control can achieve excellent TSTR by doing
precisely the wrong thing, namely memorising training rows, and the cases where
synthesis is most valuable, such as rare and high-risk patients, are the cases
where memorisation is most likely. Under UK GDPR and the ICO's anonymisation
guidance, a release is defensible only if re-identification risk is assessed and
bounded. Differential privacy is the strongest available form of that bound: a
worst-case, composable, attacker-agnostic limit on what any adversary can learn
about any single record's presence.

The privacy pillar is deliberately first after the baseline, for two reasons of
experimental design. Differential privacy is the safeguard that most constrains
utility, and its cost must be on the table before bias mitigation, which will
reallocate utility across subgroups, is layered on top; otherwise the two effects
become unattributable. Furthermore, DP noise is known to fall hardest on minority
subgroups, so the fairness questions of Pillar 2 are only well posed given a fixed
privacy budget. The ladder's order encodes the causal chain the thesis intends to
measure.

## 2. Threat model

**Adversary goal.** Membership inference: to decide whether a target individual's
record was in the generator's training set. Membership is the canonical atomic
leak. It is the guarantee differential privacy speaks to directly, and stronger
attacks such as attribute inference and reconstruction imply it.

**Adversary access.** Black-box, meaning the released synthetic dataset only, with
no model weights, no query access and no training-loss oracle. This matches the
intended release mode, which is a data file rather than an API.

**Adversary knowledge.** The target record's full feature vector, together with
general population knowledge, so that the adversary may fit models to public or
held-out data.

**Protected unit.** One row: an ICU stay for the tabular tasks, a calendar day for
the count series. Patient-level protection, which matters because a patient may
contribute several stays, requires group privacy and is outstanding work, discussed
in Section 8.

## 3. Mechanism: DP-SGD on the critic, accounted end to end

### 3.1 Why the critic is the right place for the noise

In CTGAN, real data enters training at exactly one point, the critic's forward
pass. The generator sees only noise, conditional vectors and the critic's
gradients. Privatising the critic's updates therefore privatises everything
downstream, including the generator's weights and every synthetic sample ever
drawn, by the post-processing property of differential privacy. This is the
architectural fact that made CTGAN the right Stage-0 platform: the privacy pillar
becomes a training-loop modification rather than a new model, so the utility
difference measured against Stage 0 is attributable to the mechanism alone.

### 3.2 DP-SGD adapted to WGAN-GP and PacGAN

Each critic step samples a batch of real rows, and rows enter the critic in PacGAN
groups. The per-group gradient of the full critic loss, comprising the Wasserstein
terms and that group's gradient penalty, is clipped in L2 norm; clipped group
gradients are summed; and Gaussian noise is added at a scale set by the
replace-one sensitivity, since a record lives in exactly one group per batch. The
generator update is unchanged from Stage 0: it touches no real data and is free
post-processing.

Two CTGAN-specific leaks are closed explicitly.

Training-by-sampling matches real rows to sampled categories, which destroys the
uniform-subsampling assumption that privacy amplification requires. In DP mode,
real rows are drawn uniformly and each row's conditional vector is derived from the
row itself, by selecting a uniformly random discrete column and setting it to that
row's own category. The critic still sees consistent row-and-condition pairs.

The conditional sampler's category frequencies are data-derived. They are released
once through a Gaussian mechanism, whose sensitivity follows from the fact that one
record shifts one bin per discrete column, and the noisy log-frequencies are then
used for both training and generation-time conditioning. That release is composed
into the same accountant.

### 3.3 Accounting

Each critic step is a sampled Gaussian mechanism. I implement the integer-order
Rényi bound of Mironov, Talwar and Zhang (2019), compose linearly over steps, add
the frequency-release term, and convert to an epsilon-delta pair by the standard
minimisation over orders. The implementation is numpy-only and MIT licensed
(`src/privacy_accountant.py`). The noise multiplier is calibrated by bisection so
that the total spend meets each target budget exactly at the configured step count,
and the spent budget is recomputed from actual steps during training and reported
per epoch. Fixed-size batches drawn without replacement are accounted as Poisson
subsampling at the same rate, which is the standard approximation in DP-SGD
practice and is stated here rather than hidden.

### 3.4 The temporal task

Bed demand's Stage-0 generator is a log-link Poisson autoregression, and its DP
counterpart is the same model fitted by DP full-batch gradient ascent. Per-row
gradients are clipped; one calendar day appears in at most a bounded number of
rows, once as a target and otherwise as a lag, which gives the replace-one
sensitivity per step; and each iteration is a Gaussian mechanism in the same
accountant. Sampling trajectories from the fitted coefficients is post-processing.
One mechanism family per data modality, each with an explicit budget, keeps the
study coherent.

### 3.5 Why not the alternatives

PATE-GAN (Jordon et al., 2019) partitions the training set among teacher
discriminators and gives the student a data-independent guarantee through noisy
vote aggregation. It is principled, but at our cohort sizes each teacher would see
a small slice of an already subsampled fit; the mechanism replaces the WGAN-GP
critic wholesale, which is a re-platforming that breaks the ladder's
attributability; and its empirical utility advantage appears at scales larger than
ours. It is the planned comparison arm, not the platform.

Output perturbation, and DP releases of low-order marginals such as MST-style
mechanisms, guarantee those marginals but cannot express the joint, conditional
structure that Stage 0 showed is precisely what downstream models need.

Ad-hoc anonymisation, meaning k-anonymity and its relatives, provides no protection
against attackers with background knowledge, does not compose, and is exactly what
differential privacy was invented to replace. It appears in this programme only in
Pillar 3, as part of the evidence package, and never as the guarantee.

## 4. Experimental design

Everything about the Stage-0 evaluation is frozen: the same derived feature
matrices, the same patient-disjoint splits, the same downstream models
(RandomForest for length of stay, XGBoost for readmission, a Poisson-AR forecaster
for bed demand) and the same real-test-set scoring. One axis is added, the privacy
budget, together with the Stage-0 numbers as the unbounded-budget reference. For
each budget I measure two things.

**Utility.** TSTR at that budget against the TRTR ceiling. The marginal cost of
privacy is the difference between the non-private and the private TSTR, cleanly
separated from the Stage-0 cost of synthesis, which is the difference between TRTR
and non-private TSTR.

**Leakage.** Two black-box membership-inference attacks with real training rows as
members and real test rows as non-members. The first scores a candidate by the
negative distance to its nearest synthetic record. The second trains a logistic
regression on distances to the first, fifth and tenth nearest synthetic neighbours,
using half the candidates for attack training and half for scoring. An attack AUC of
0.5 indicates no leakage, and near-duplicate statistics report memorisation
directly.

I registered three hypotheses before running the experiments. H1: utility falls
monotonically as the budget shrinks, with the readmission minority class hit
hardest, this being the structure Stage 0 showed is most fragile. H2: the
non-private release shows attack AUC measurably above 0.5, and differential privacy
pulls it to approximately 0.5 across the grid; if instead the non-private attacks
already sit at chance, the interesting finding is that the formal bound, rather than
the empirical attack, is what justifies the budget. H3: bed demand, whose sufficient
statistics are low-dimensional, is nearly budget-insensitive at practical settings.

## 5. Results

Authoritative figures are the five-seed means with 95% confidence intervals from
the companion repository `mimic-synthetic-evidence`. Accounting was audited: every
dataset spent within 0.01 of its target budget.

| Task | Metric | TRTR | Non-private | DP at release budget |
|---|---|---|---|---|
| Length of stay | R² | 0.186 ± 0.001 | 0.009 ± 0.083 | -0.671 ± 1.481 |
| Readmission | ROC-AUC | 0.778 ± 0.007 | 0.690 ± 0.018 | 0.519 ± 0.085 |
| Readmission | PR-AUC | 0.379 ± 0.014 | 0.246 ± 0.020 | 0.152 ± 0.035 |
| Bed demand | MAE | 1.475 ± 0.000 | 1.475 ± 0.005 | 1.460 ± 0.005 |
| Membership-inference AUC (all arms) | | n/a | 0.497 to 0.505 ± 0.006 | 0.497 to 0.505 ± 0.006 |

### 5.1 Hypothesis verdicts

**H1 is partially supported.** The cost of privacy on readmission is significant
(p = 0.005). On length of stay, R² is negative at every budget, meaning the
DP-synthetic-trained regressor is worse than predicting the mean, but the mean cost
is not statistically resolvable at five seeds, for the reason given in Section 5.3.
The minority class is hit hardest, as predicted, with PR-AUC falling further in
relative terms than ROC-AUC. A concrete secondary observation: the degenerate
accuracy figures observed in the single-seed grid trace to DP distortion of the
synthetic label prevalence, which mis-sets XGBoost's positive-class weighting. That
is a reportable interaction between DP generation and downstream class-imbalance
handling.

**H2 resolves to its alternative branch, and this is the more interesting outcome.**
Both black-box attacks sit at chance at every budget, including the non-private
baseline, with all attack cells falling between 0.497 and 0.505, and with no
near-duplicate synthetic records. The non-private generator, at this fidelity, does
not memorise detectably. It follows that empirical black-box membership inference
cannot distinguish the tightest budget from the unbounded one, and therefore cannot
certify any release. The justification for a release budget is the formal
epsilon-delta bound; the attack suite is a falsification check, not the guarantee.
Stronger adversaries, whether white-box, shadow-model-calibrated, or
attribute-inference, remain future work and could still separate the budgets. This
inverts the common practice of citing a failed attack as evidence of safety.

**H3 is confirmed.** Bed demand's MAE stays within 0.02 of the TRTR ceiling at every
budget, even at a very large noise multiplier. A two-parameter sufficient-statistic
model is intrinsically robust to gradient noise.

### 5.2 The frontier in one sentence

Privacy is nearly free for low-dimensional aggregate structure and, at this
generator class and compute budget, extremely expensive for high-dimensional joint
structure.

### 5.3 Differential privacy destroys predictability

The multi-seed re-run produced a finding that no single training run can reveal.
Across seeds, differential privacy inflates the generator's run-to-run standard
deviation by a factor of 4.8 on readmission ROC-AUC (from 0.014 to 0.068) and by a
factor of 17.9 on length-of-stay R² (from 0.066 to 1.193). Individual runs at the
same release budget produced R² values ranging from -0.041 to -2.799.

Two consequences follow, and both are material.

In practice, an unpredictable generator is arguably worse than a predictably poor
one. A single run cannot tell a data controller whether the dataset they are about
to release is usable, and that is exactly the decision they face. An unpredictable
harm is also harder to govern than a systematic one, because a systematic one can
be corrected.

Methodologically, the mean utility cost of privacy on length of stay is not
statistically resolvable at five seeds (p = 0.28), not because the cost is small
but because the variance is enormous. Reporting a single DP training run as the
cost of privacy, which is routine in this literature and which the first version of
this paper did, is therefore unsound. Every claim in this programme is now checked
against a measured noise floor before it is reported.

### 5.4 Correction

The first version of this study used a row-level readmission split, which placed
roughly 43% of the test set in the hands of patients the model had already seen.
For utility the effect was small, about 0.011 ROC-AUC on the ceiling. For the
membership-inference evaluation it was fatal, because the non-members were not
non-members. The split is now patient-disjoint and the attack results in Section 5.1
are measured against a valid control group. The headline privacy finding survives
that correction, which is worth stating plainly: it was the finding most at risk
from the defect.

## 6. What the guarantee covers, and what it does not

**Covered.** The critic's entire training trajectory; the conditional-frequency
release; and, for bed demand, the fitted autoregressive coefficients. Hence, by
post-processing, the generator weights and all synthetic samples.

**Not covered, and documented.** The per-column variational Gaussian-mixture
normalisation and the discrete category schema are fitted on real training data
before training begins; a DP quantile normalisation is the planned refinement. The
subsampling approximation of Section 3.3 stands. Adjacency is row-level rather than
patient-level.

**Empirical results are not proofs.** The attack suite checks that the bound does
practical work; it cannot substitute for it.

## 7. Relation to the four-pillar ladder

Stage 1 fixes the privacy mechanism and the budget, and every later pillar inherits
both. Pillar 2 modifies how the conditional column and its frequencies are chosen,
which is why those frequencies were given their own accounted release here, and it
asks whether subgroup disparity can fall at a fixed budget. Pillar 3 packages
Section 6 into a data protection impact assessment with model and data cards mapped
to ICO, EU AI Act and GDPR expectations. Pillar 4 attempts to close whatever TSTR
gap remains, under the same budget, so that the final comparison is
utility-at-fixed-privacy.

## 8. Future work

Patient-level adjacency, through group privacy or per-patient Poisson sampling.
PATE-GAN and DP diffusion comparison arms. DP-fitted normalisation statistics.
Attribute-inference and reconstruction attacks to complement membership inference.
A formal audit of the accountant against reference implementations. And, given
Section 5.3, an investigation of whether ensembling across independent DP runs can
buy back predictability, and at what cost in budget.

---

### References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and
Zhang, L. (2016). *Deep Learning with Differential Privacy.* ACM CCS.

Mironov, I., Talwar, K., and Zhang, L. (2019). *Rényi Differential Privacy of the
Sampled Gaussian Mechanism.* arXiv:1908.10530.

Mironov, I. (2017). *Rényi Differential Privacy.* IEEE CSF.

Xie, L., Lin, K., Wang, S., Wang, F., and Zhou, J. (2018). *Differentially Private
Generative Adversarial Network.* arXiv:1802.06739.

Jordon, J., Yoon, J., and van der Schaar, M. (2019). *PATE-GAN: Generating
Synthetic Data with Differential Privacy Guarantees.* ICLR.

Xu, L., Skoularidou, M., Cuesta-Infante, A., and Veeramachaneni, K. (2019).
*Modeling Tabular Data using Conditional GAN.* NeurIPS.

Chen, D., Yu, N., Zhang, Y., and Fritz, M. (2020). *GAN-Leaks: A Taxonomy of
Membership Inference Attacks against Generative Models.* ACM CCS.

Johnson, A., Bulgarelli, L., Pollard, T., et al. (2024). *MIMIC-IV* (v3.1).
PhysioNet.
