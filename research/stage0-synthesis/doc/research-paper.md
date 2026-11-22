# A Conditional-GAN Baseline for Synthetic Electronic Health Records, and a Four-Pillar Evaluation Programme

Belhassen Ouerghi. Professional Engineering Doctorate, University of Hertfordshire.

Working paper. Stage 0 (utility baseline) of the thesis *Advanced Data Generation
for Healthcare Predictive Modelling: Ensuring Privacy, High Realism, Bias
Mitigation and Ethical Compliance.*

---

## Abstract

Synthetic data is increasingly proposed as a route to sharing the analytic value
of electronic health records without exposing the patients behind them. Its
usefulness is conditional. A synthetic dataset is only worth generating if a model
trained on it still performs on real data, and only worth releasing if it does not
leak the individuals it was derived from. This paper establishes the utility
baseline for a doctoral programme that studies that trade-off end to end. I select
CTGAN (Conditional Tabular GAN) as the base generator, implement it from first
principles, and fit it to the real patient-level cohorts underlying three
previously replicated MIMIC-IV prediction models: ICU length of stay
(regression), 30-day readmission (classification) and daily ICU bed demand
(temporal forecasting). Utility is measured with the standard
train-on-synthetic, test-on-real (TSTR) protocol against a train-on-real,
test-on-real (TRTR) ceiling. The result is a reference point, namely how much
predictive signal survives synthesis before any privacy, fairness, ethical or
realism intervention is applied, against which each of the four thesis pillars is
subsequently measured. I argue why CTGAN in particular is the right scaffold for
an incremental programme of this kind, and set out the experimental ladder the
remaining pillars climb.

---

## 1. Motivation and scope

The three companion repositories established that routine first-day ICU data
carries real predictive signal on MIMIC-IV. A length-of-stay model reproduces the
upstream patient-level R² almost exactly, a 30-day failed-discharge classifier
reaches a ROC-AUC of about 0.78, and daily ICU admissions can be forecast with a
Generalized-Poisson autoregressive model. Those are models trained and tested on
real data.

The doctoral question is what happens when the training data is synthetic. If a
generator can produce a dataset that trains an equally good model, and that can
later be made provably private, fair, compliant and realistic, then it is that
dataset, rather than the real one, that a hospital could share or publish. Stage 0
answers only the first half of the question: the raw utility of synthetic EHR
before any pillar is applied. This is deliberately the most permissive setting.
Every later pillar can only cost utility, so the number established here is the
ceiling for the whole programme.

The scope boundaries are worth stating. I do not, at this stage, apply
differential privacy, measure membership-inference risk, audit subgroup fairness,
or attempt realism enhancements; those are Pillars 1 to 4, described in Section 6.
Nor do I claim the synthetic data is safe to release. Section 7 explains why it is
instead treated with the same caution as real MIMIC-IV data until the privacy
pillar has been evaluated.

## 2. Data

All real data are the derived, de-identified feature matrices produced by the
three replication repositories from MIMIC-IV v3.1. Raw MIMIC is never re-extracted
here.

| Task | Grain | Rows (train / test) | Features | Target |
|---|---|---|---|---|
| Length of stay | ICU stay | 79,693 / 14,037 | 29 (physiology and demographics) | truncated LOS in days (regression) |
| Readmission | ICU stay | 69,699 / 17,342 | 72 (physiology and utilisation) | 30-day failed discharge (binary) |
| Bed demand | day | 3,289 / 364 | n/a | daily ICU-admission count (temporal) |

The length-of-stay split reuses the companion repository's saved patient-level
split. The readmission split is a patient-disjoint hold-out generated with
`GroupShuffleSplit` on the patient identifier; an earlier version of this study
used a row-level split, which leaked patients across the boundary, and Section 5.2
records the correction. In both cases the real test set is withheld entirely from
synthesis, which is what makes the TSTR comparison valid.

## 3. Why CTGAN: the argument for the base generator

Choosing the base generator is the load-bearing decision for a four-pillar
programme, because every pillar is implemented as a modification of the generator
or of its training objective. The choice is therefore made not only on baseline
fidelity but on how cleanly each future pillar attaches. I select CTGAN (Xu et
al., 2019) for five reasons.

**It is designed for this data shape.** MIMIC feature matrices are mixed-type:
continuous, often multimodal and non-Gaussian lab and vital distributions sit
alongside categorical demographics with severe class imbalance. CTGAN's two
defining mechanisms target precisely these pathologies. Mode-specific
normalisation represents each continuous column as a mixture of Gaussians, so that
a bimodal creatinine distribution is not crushed to a single mean, and
training-by-sampling with a conditional vector prevents rare categories from being
ignored by the generator. A plain GAN, a Gaussian copula, or a VAE with a
unimodal Gaussian latent all handle this worse.

**The conditional mechanism is the fairness pillar's natural hook.** Pillar 2
requires conditional generation on sensitive attributes, in order to resample or
reweight the synthetic distribution over subgroups. CTGAN already conditions
generation on a chosen discrete column by construction, so the bias pillar becomes
a change to how the conditional column and its sampling frequencies are chosen,
rather than a new architecture. Committing to a conditional generator now avoids
re-platforming later.

**The discriminator objective is the privacy pillar's natural hook.** Pillar 1 is
most cleanly realised by training the same GAN under differential privacy, either
with DP-SGD on the discriminator or with a PATE-aggregated teacher ensemble. Both
are drop-in modifications of CTGAN's critic training loop, which is why they exist
as CTGAN variants in the literature. A non-adversarial generator would not admit
this family of guarantees so directly.

**It is the field's reference baseline.** The thesis contributes a
privacy-fidelity-fairness benchmark across generator families. CTGAN is the
single most-cited tabular synthesiser and the default baseline in the
tabular-synthesis and health-data-synthesis literature. Anchoring Stage 0 on it
makes every later comparison legible to that literature.

**It can be implemented licence-clean and fully controlled.** The programme
requires permissive-licence-only code, and the widely used `sdv` and `ctgan`
packages are Business-Source Licensed. Because CTGAN's mechanisms are
well-specified, I re-implemented it from scratch in `src/ctgan.py` under MIT. This
additionally makes every mechanism a pillar will touch, such as the conditional
vector and the critic loss, explicit and editable rather than buried in a
dependency.

### 3.1 What CTGAN is not, and why that is acceptable

CTGAN is not the highest-fidelity tabular generator available; diffusion-based
methods generally lead on marginal and joint fidelity. That is by design. Realism
is Pillar 4, the final step, not the starting point. Starting from a mid-fidelity
but maximally extensible baseline, and earning realism last, is the correct
ordering for a study whose subject is trade-offs, because it leaves head-room in
which the realism intervention can be shown to move the needle or fail to.

## 4. Method

### 4.1 The generator

I implement CTGAN faithfully: per-continuous-column variational Gaussian mixtures
(`BayesianGaussianMixture` with a Dirichlet-process prior) for mode-specific
normalisation; one-hot encoding for categoricals; a conditional generator with
training-by-sampling of the conditioned discrete column, where categories are
drawn in proportion to log-frequency so that minorities are seen; a WGAN-GP critic
with gradient penalty and PacGAN packing; and the conditional cross-entropy term
that forces the generated conditioned column to match the requested category.
Continuous outputs use a `tanh` scalar plus a Gumbel-softmax mode indicator, and
categorical outputs use Gumbel-softmax. Full hyper-parameters are in
`config.yaml`; the headline settings are 200 epochs, batch size 500, embedding
dimension 128, and generator and critic hidden layers of 256 by 256. Per-epoch
generator and critic losses are persisted to `results/train_log.txt` so that
convergence is on record. To respect a CPU budget the GAN is fitted on a
30,000-row subsample of each training split, then sampled to the full
training-set size so that the downstream comparison uses equally sized real and
synthetic training sets.

### 4.2 The temporal counterpart

CTGAN treats rows as exchangeable and so cannot synthesise a single autocorrelated
count series; shuffling days would destroy exactly the signal the forecaster
needs. For bed demand I therefore use the modality-appropriate generative model, a
Poisson autoregressive process fitted to the real training span, from which a
synthetic trajectory of equal length is sampled. This keeps the study coherent,
with one generative model per data modality, and it is stated plainly as a scope
decision. A dedicated deep time-series synthesiser, such as TimeGAN, is noted as
future work.

### 4.3 The evaluation protocol

Utility is measured by TSTR against TRTR. For each task a single downstream model
is fixed and only its training set changes, real for TRTR and synthetic for TSTR,
with both scored on the same untouched real test set. Length of stay uses a
`RandomForestRegressor` scored by RMSE, MAE and R². Readmission uses an XGBoost
classifier scored by ROC-AUC, PR-AUC and accuracy. Bed demand uses a Poisson-AR
forecaster scored by MAE and RMSE on the real test span. A small gap between the
two arms is direct evidence that the synthetic data preserved the predictive
signal; a large gap quantifies the utility cost of synthesis. Because the real
test set is never seen by the generator, TSTR cannot be inflated by leakage.

## 5. Results

### 5.1 Headline figures

The authoritative numbers come from the five-seed re-run on patient-disjoint
splits (companion repository `mimic-synthetic-evidence`), reported as mean plus or
minus a 95% confidence interval.

| Task (model) | Metric | TRTR | TSTR |
|---|---|---|---|
| Length of stay (RandomForest) | R² | 0.186 ± 0.001 | 0.009 ± 0.083 |
| Readmission (XGBoost) | ROC-AUC | 0.778 ± 0.007 | 0.690 ± 0.018 |
| Readmission (XGBoost) | PR-AUC | 0.379 ± 0.014 | 0.246 ± 0.020 |
| Bed demand (Poisson-AR) | MAE | 1.475 ± 0.000 | 1.475 ± 0.005 |

The cost of synthesis is statistically significant on both tabular tasks (paired
t-test across seeds: readmission p = 0.0002, length of stay p = 0.004).

The three tasks span the full range of outcomes, and the differences between them
are themselves informative. Bed demand survives synthesis essentially intact. The
Poisson-AR sampler preserves exactly the low-order temporal statistics, the mean
level and the first-order autocorrelation, that the downstream forecaster
consumes; where generator and forecaster share an inductive bias, utility
transfers almost losslessly. Readmission retains most of its signal, with a TSTR
ROC-AUC of 0.690 keeping roughly two-thirds of the above-chance discrimination,
while the sharper fall in PR-AUC indicates that the rare-positive conditional
structure, which is precisely what training-by-sampling is meant to protect,
remains the first casualty. Length of stay is the most severe case: R² falls to a
value indistinguishable from zero. The synthetic cohort reproduces the marginal
distributions well (Figure `fidelity_los.png`) but carries little of the
cross-feature conditional structure the RandomForest exploits. Marginals preserved
with joint dependencies lost is the canonical signature of a mid-fidelity tabular
GAN under a constrained compute budget, and it fixes the head-room that Pillar 4
must recover.

The bed-demand result requires one caution, because it is easy to misread. TSTR
error is occasionally lower than TRTR, and this must not be read as synthetic data
outperforming real data. It is a tie within noise. The TSTR arm is a parametric
bootstrap: the synthetic series is sampled from the model fitted on the real
series, and refitting recovers nearly identical coefficients. The two forecasts
never diverge by more than 0.05 admissions per day over the 364-day horizon, and
the sign of the difference reverses under RMSE. By the data-processing inequality
a randomised function of the training data cannot carry more information about the
test period than the training data itself. Apparent cases of TSTR matching or
exceeding TRTR arise only from sampling noise or from implicit regularisation,
here a mild shrinkage of the autoregressive term that MAE rewards and RMSE
penalises.

### 5.2 Corrections to the first version of this paper

Two defects in the original study were found and corrected, and they are recorded
here rather than removed.

The first was data leakage. The readmission cohort holds 87,041 ICU stays from
only 60,537 patients, and the original hold-out was a row-level stratified split,
which placed 6,046 patients on both sides of it; roughly 43% of the test set
consisted of patients the model had already seen. The split is now patient-disjoint.
The effect on the TRTR ceiling proved small, about 0.011 ROC-AUC, but the same
defect invalidated the membership-inference evaluation reported in Stage 1, whose
non-members must be patients the generator never saw. That is a validity problem
rather than a magnitude problem, and it could not be excused by the small effect
size.

The second was a reporting error. An earlier version of this section argued that
doubling the training budget lifted LOS R² from 0.009 to 0.054, and inferred from
this that part of the TSTR gap was optimisation rather than a ceiling. The claim
does not survive replication. Across five seeds at the same 200-epoch budget, LOS
TSTR R² takes the values 0.054, 0.038, 0.006, -0.104 and 0.053, giving a mean of
0.009 with a 95% confidence interval of ±0.083. The reported figure was the most
favourable of five runs. Whether a larger training budget improves length-of-stay
synthesis is unresolved; the defensible claim is that synthesis destroys
essentially all of the length-of-stay signal. The error is instructive, and it is
the reason every headline number in this programme now carries a confidence
interval and is checked against a measured noise floor before it is reported.

## 6. The four-pillar programme

Stage 0 fixes the generator and the evaluation harness. The thesis contribution is
the incremental study that follows. Each pillar is a modification applied on top
of the previous one, generating a new dataset and re-running the identical
TSTR/TRTR comparison, so that the utility cost of each safeguard is isolated. The
five dataset generations are versioned on disk accordingly.

| Stage | Dataset folder | Intervention added | Primary question |
|---|---|---|---|
| 0 (this paper) | `MIMIC-Synthetic` | CTGAN base | How much utility survives synthesis? |
| 1 | `MIMIC-SyntheticPrivacy` | Differential privacy (DP-SGD critic) | What utility does a formal epsilon budget cost, and does it defeat membership-inference attacks? |
| 2 | `MIMIC-SyntheticPrivacyBias` | Bias mitigation (fairness-aware conditional generation) | Can subgroup disparity be reduced without further utility loss, and does DP noise fall hardest on minorities? |
| 3 | `MIMIC-SyntheticPrivacyBiasEthic` | Ethical and regulatory compliance | What evidence package makes a release defensibly anonymous and audit-ready? |
| 4 | `MIMIC-SyntheticPrivacyBiasEthicRealism` | Realism enhancement | Can realism be recovered after the safeguards, closing the TSTR gap? |

At every stage the same three prediction tasks are re-evaluated, producing a matrix
of utility, privacy, fairness, compliance and realism across five dataset
generations. That matrix is the empirical spine of the thesis. Privacy comes first
after the baseline because it is the pillar that most constrains utility, and
because no synthetic EHR should be released before it is applied, which is exactly
why the Stage-0 data here remains unreleased.

## 7. Data governance and ethics

This work is conducted under the PhysioNet Credentialed Health Data Use Agreement
for MIMIC-IV v3.1. No raw or derived real patient-level data is committed to the
repository. The synthetic data is also not committed: a synthetic dataset that has
not yet passed the privacy pillar has no formal privacy guarantee and may retain
re-identification risk, which is the very risk Pillar 1 exists to measure and
bound. It is therefore written outside the repository and treated with the same
caution as real data until proven otherwise. Only source code, aggregate metrics
and aggregate figures are version-controlled. The models here are research
artefacts, not validated medical devices, and must not inform clinical care.

## 8. Conclusion

I have established the utility baseline for a four-pillar synthetic-EHR
programme: a from-scratch, licence-clean CTGAN applied to three real MIMIC-IV
prediction tasks and evaluated by TSTR against a TRTR ceiling. The choice of CTGAN
is justified not by its baseline fidelity but by how directly its conditional
generator and adversarial critic host the privacy, bias, ethics and realism
interventions to come. The measured baseline is sobering: on length of stay,
synthesis alone destroys essentially all predictive signal before privacy is even
applied, which means the realism pillar must repair the base generator and not
merely the privacy layer. The next stage introduces differential privacy and the
first membership-inference evaluation.

---

### References

Xu, L., Skoularidou, M., Cuesta-Infante, A., and Veeramachaneni, K. (2019).
*Modeling Tabular Data using Conditional GAN.* NeurIPS. arXiv:1907.00503.

Jordon, J., Yoon, J., and van der Schaar, M. (2019). *PATE-GAN: Generating
Synthetic Data with Differential Privacy Guarantees.* ICLR.

Xie, L., Lin, K., Wang, S., Wang, F., and Zhou, J. (2018). *Differentially Private
Generative Adversarial Network.* arXiv:1802.06739.

van Breugel, B., Kyono, T., Berrevoets, J., and van der Schaar, M. (2021).
*DECAF: Generating Fair Synthetic Data Using Causally-Aware GANs.* NeurIPS.

Johnson, A., Bulgarelli, L., Pollard, T., et al. (2024). *MIMIC-IV* (v3.1).
PhysioNet.

Companion replications: `Length-of-Stay-Prediction-Replica`,
`readmission-prediction-replica`, `bed-demand-prediction-replica`.
