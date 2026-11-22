# Fair Synthetic Electronic Health Records at a Fixed Privacy Budget: the Bias Pillar of a Four-Pillar Programme

Belhassen Ouerghi. Independent research, part of [SynthEHR](../../../README.md).

Working paper. Stage 2 (bias pillar) of SynthEHR, an independent research programme on synthetic health data.

---

## Abstract

Stage 1 of this programme priced a formal differential-privacy guarantee on synthetic
EHR and found its cost concentrated where the fairness literature predicts, in rare
and minority-class structure. This paper adds the bias pillar. At the fixed release
budget I retrain the DP generator and layer on two bias-mitigation arms that exploit
the conditional mechanism for which CTGAN was originally chosen. Fair conditional
sampling re-weights the generation-time conditioning toward a balanced target over
sensitive subgroups; because it is post-processing of a differentially private
generator, it consumes no additional privacy budget. Fair training-by-sampling
replaces the frequency weights of the sensitive columns with a uniform distribution
during training, a data-independent change that is likewise free of privacy cost.
Every arm, together with the real-data, non-private and DP-unmitigated references, is
scored on the same real test sets for utility and for subgroup fairness.

The stage's principal contribution is not the one I set out to make. The first
version of this study reported that differential privacy systematically collapses
subgroup representation, and that fair sampling repairs it. Replication across five
seeds retracts that claim: the collapse was a single seed. What replaces it is a
stronger and more general finding, consistent with Stage 1: differential privacy does
not bias representation in a fixed direction, it makes representation unstable. I
also report a measured noise floor which shows that the utility effects of the
fairness interventions are, at this budget, too small to be measurable at all.

---

## 1. Motivation

A synthetic EHR release can be unfair in two distinct ways. It can misrepresent the
population, because generative models under-produce minority subgroups and the groups
that healthcare serves worst arrive at the generator already under-documented. And it
can transfer disparate utility, so that a model trained on the release performs worse
for some groups on real patients.

Differential privacy sharpens both concerns, because clipping and noising suppress
precisely the gradient signal that rare subgroups contribute (Bagdasaryan et al.,
2019). Stage 1's results exhibit this: at every budget the minority readmission class
lost the most in relative terms. In the readmission cohort the stakes are not
hypothetical, since Medicare patients are readmitted at 17.1% against 9.1% for the
privately insured, while Medicaid patients make up 15.5% of the cohort.

The ladder's order is deliberate. Fairness questions are only well posed at a fixed
privacy budget, which is why Pillar 1 comes first, and the compliance pillar will need
the disparity evidence this stage produces. One cannot mitigate a harm before the
mechanism that causes it exists.

## 2. Mechanism

### 2.1 Fair conditional sampling: generation-time, and free in budget

CTGAN's conditional cross-entropy forces the generator to honour the conditional
vector, so the subgroup composition of a released sample is a sampling-time choice.
`sample_fair()` conditions every draw on a sensitive column, drawing the category from
a target distribution that is uniform over adequately supported categories, with rare
categories kept at their observed share rather than inflated. A category with 0.03%
support must not become a fifth of a synthetic population that the generator has never
meaningfully modelled.

Because the generator was trained under differential privacy, any function of it,
including re-weighted sampling, inherits the guarantee by post-processing. The fairness
intervention is therefore free in privacy budget. The targets are computed from the
same noisy-count release that Stage 1 already accounts for, so no new real-data
statistic is consumed. This argument is theoretical and does not depend on the data,
which, as Section 4 will show, matters a great deal.

### 2.2 Fair training-by-sampling: training-time, also free

Standard training-by-sampling draws the conditioned category in proportion to noisy
log-frequency, so the critic sees minority categories rarely. The fair-training arm
replaces those weights, for the sensitive columns only, with a uniform distribution.
This is a data-independent constant and costs nothing in the accountant. Minority
subgroups are then seen as often as majority ones during fitting. Where fair sampling
changes minority count, fair training targets minority fidelity.

### 2.3 Why not the alternatives

Causally aware fair generation, as in DECAF, needs a defensible causal graph over 29 to
72 clinical features, which does not exist. Pre-processing repair of the real data
breaks the privacy accounting chain at its root. Post-hoc relabelling of synthetic
outcomes fabricates clinical ground truth. Re-weighting the conditional mechanism the
platform already has is architecture-native, free in budget, and, because everything
else is frozen, attributable, which is the ladder's design invariant.

## 3. Experimental design

Frozen from Stages 0 and 1: the data, the patient-disjoint splits, the real test sets,
the downstream models, the TSTR/TRTR protocol, the architecture, and the privacy
accounting at the release budget. Sensitive attributes are gender, on both tabular
tasks, and insurance, on readmission. Six arms per task are placed on the
fairness-utility plane: TRTR, the non-private Stage-0 release, the DP-unmitigated
Stage-1 release, and fresh `dpbaseline`, `fairsample` and `fairtrain` runs. Bed demand
is a DP pass-through, since a count series has no sensitive attribute; this is a scope
statement, not an omission.

Fairness metrics are computed on the real test set for groups with at least 200
members: per-group ROC-AUC and its maximum-minus-minimum gap; the demographic-parity
difference, being the positive-rate gap at the default threshold; the equalized-odds
difference, being the larger of the true-positive-rate and false-positive-rate gaps
(Hardt et al., 2016); for regression, the per-group RMSE gap and the mean-signed-error
gap; and representation shares against the real population. The released arm is
re-attacked with the Stage-1 membership-inference suite.

I registered three hypotheses and one caution before running the experiments. H1: fair
sampling restores near-balanced representation and narrows disparity gaps relative to
the unmitigated DP baseline, at indistinguishable utility, since it samples the same
generator. H2: fair training additionally improves minority-subgroup AUC, possibly at
some cost to majority utility. H3: disparity increases along the ordering from real
data, through non-private synthesis, to differentially private synthesis.

The caution: Stage 1 measured readmission utility near chance at this budget, and a
chance-level classifier can be trivially fair, because its errors are uniformly
distributed. Disparity results would therefore be interpreted only jointly with
utility. If the fair arms showed narrowed gaps at near-zero utility, the honest
conclusion would be that fairness at a fixed budget is vacuous at this generator class,
and that the realism pillar must restore utility before the fairness result can mean
anything.

## 4. Results

Authoritative figures are five-seed means with 95% confidence intervals from the
companion repository `research/evidence`. All arms spent within 0.01 of the
target budget.

### 4.1 The noise floor

Re-running the identical DP recipe on the identical split, changing only the
generator's random initialisation, moves readmission ROC-AUC by a mean of 0.095
(maximum 0.157) and length-of-stay R² by a mean of 1.083 (maximum 3.756). The minimum
detectable effect at five seeds is therefore approximately 0.151 in ROC-AUC and 3.08 in
R².

The fairness interventions produce effects of 0.035 to 0.041 in ROC-AUC. They are three
to four times smaller than the noise against which they must be measured. Resolving them
would require roughly 50 seeds on readmission and thousands on length of stay. This is a
property of DP generative modelling at this budget rather than a defect of the study,
and it has a corollary that indicts standard practice: reporting single-run DP results,
which is the norm in this literature, is unsound. It is also the error the first version
of this paper made, as the next section records.

### 4.2 The retraction

The first version of Section 4 reported that differential privacy collapsed subgroup
representation, driving the Medicare share from 54.1% of the real cohort to 5.9% of the
synthetic one, and called this a systematic harm channel. **That claim is retracted.**

Across five seeds, the unmitigated DP generator produces Medicare shares of 0.421,
0.095, 0.285, 0.613 and 0.618, against a real share of 0.542 and a non-private synthetic
share of 0.524 ± 0.080. The mean is 0.406 with a standard deviation of 0.223.
Differential privacy ranges from near-total collapse to nearly correct. The figure of
5.9% was the tail of a distribution, reported as though it were the distribution.

### 4.3 The corrected finding

Differential privacy at practical budgets does not bias representation in a fixed
direction. It makes representation unstable: the run-to-run standard deviation of the
Medicare share is inflated by a factor of 2.8, from 0.080 to 0.223. This is the same
phenomenon that Stage 1's utility metrics show, with run-to-run standard deviation
inflated by factors of 4.8 and 17.9, now surfacing in a second and independent
measurement.

The practical consequence is sharper than the claim it replaces. A data controller who
runs the generator once cannot know whether the cohort they are about to release
under-represents its sickest subgroup by a factor of six or reproduces it faithfully,
and no amount of auditing of that single run will tell them which world they are in. An
unpredictable harm is worse than a systematic one, because a systematic one can be
corrected.

### 4.4 The interventions, reassessed

Fair sampling is not repairing a collapse, because there is no systematic collapse. It
moves the distribution toward its uniform target, which is to say it balances rather
than restores, and in the original seed this happened to look like repair. It does
measurably stabilise representation, reducing the standard deviation from 0.223 to
0.149. The claim that it costs no additional privacy budget stands untouched, because it
is theoretical and never depended on the data.

The claim that it costs no utility is downgraded. The measured effect lies far below the
detection threshold of Section 4.1, so the honest statement is that no cost is
detectable, not that there is none.

The fair-training verdict is likewise retracted. The reported collapse, at R² = -3.72,
was the worst of five runs; across seeds the gap to fair sampling is within noise
(p = 0.52). The correct claim is more interesting than the one it replaces: at this
budget the DP generator's instability is large enough to swamp the difference between
training-time and sampling-time fairness interventions. The instability dominates the
intervention, which is itself a result about the feasibility of tuning fairness under
differential privacy, and a direct argument for the realism pillar.

### 4.5 Hypothesis verdicts, and the vacuity caution

H1 is not established: representation is stabilised rather than repaired, and the
utility component is unmeasurable. H2 is not established, for the same reason. H3 is
confounded on readmission by the utility floor.

The caution registered in Section 3 proved warranted. The fair arms exhibit
demographic-parity and equalized-odds differences of exactly zero, and verification shows
why: the classifiers trained on those arms predict positive for nobody, so per-group
positive rates are all zero. A useless model is trivially fair. This is precisely the
failure mode that was anticipated, and it is why disparity was pre-committed to joint
interpretation with utility. Had the parity table been reported alone, this study would
have published a flattering artefact.

### 4.6 Privacy re-check

The released arm attacks at chance, with attack AUC between 0.497 and 0.505 and no
near-duplicate records. Fair conditioning re-weights which regions of the generator's
distribution are sampled and does not reopen a memorisation channel.

## 5. What the guarantee covers

Unchanged from Stage 1. The budget covers the critic's training trajectory and the
noisy-frequency release; the fairness interventions are post-processing or
data-independent re-weighting and add nothing to it. The Stage-1 caveats, concerning the
normalisation statistics, the category schema, row-level adjacency and the subsampling
approximation, carry over unchanged and remain in the refinement queue for Pillar 3's
evidence package.

## 6. Relation to the ladder

Pillar 3 packages this stage's disparity evidence, Stage 1's privacy evidence, and the
documented gaps in the guarantee into an impact assessment with model and data cards,
mapped to ICO, EU AI Act and GDPR expectations. Pillar 4 then attempts to raise the
utility of the fair, private generator under the same budget and the same fairness
targets. Its success is a precondition for this stage's fairness results meaning
anything clinically, which is a prediction the ladder's ordering makes and which Pillar 4
will test directly.

---

### References

Hardt, M., Price, E., and Srebro, N. (2016). *Equality of Opportunity in Supervised
Learning.* NeurIPS.

Bagdasaryan, E., Poursaeed, O., and Shmatikov, V. (2019). *Differential Privacy Has
Disparate Impact on Model Accuracy.* NeurIPS.

van Breugel, B., Kyono, T., Berrevoets, J., and van der Schaar, M. (2021). *DECAF:
Generating Fair Synthetic Data Using Causally-Aware GANs.* NeurIPS.

Xu, L., Skoularidou, M., Cuesta-Infante, A., and Veeramachaneni, K. (2019). *Modeling
Tabular Data using Conditional GAN.* NeurIPS.

Abadi, M., et al. (2016). *Deep Learning with Differential Privacy.* ACM CCS.

Mironov, I., Talwar, K., and Zhang, L. (2019). *Rényi Differential Privacy of the Sampled
Gaussian Mechanism.* arXiv:1908.10530.

Johnson, A., Bulgarelli, L., Pollard, T., et al. (2024). *MIMIC-IV* (v3.1). PhysioNet.
