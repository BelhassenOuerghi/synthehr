# Realism Under Privacy: the Fourth Pillar of a Synthetic-EHR Programme

*Belhassen Ouerghi, EngD, University of Hertfordshire.*

*Sections 1 to 4 were written before the main experiment's results existed; Sections 5
onwards report them. The pre-registration is [`PREREGISTRATION.md`](PREREGISTRATION.md).*

## Abstract

The first three pillars of this programme showed that synthesis with CTGAN and
differential privacy at ε = 10 together destroy almost all the predictive value of
MIMIC-IV feature data for 30-day readmission and ICU length of stay. This pillar asks
whether that value can be recovered inside the same safeguards. I implement TabDDPM
from scratch with a differentially private trainer (true Poisson subsampling,
per-example gradients, noise multiplicity, and an optional patient-level guarantee)
and compare it with the programme's CTGAN and DP-CTGAN in a pre-registered study: 30
arms per task, five seeds on patient-disjoint splits, independent replicates for a
noise floor, Holm-adjusted paired tests, and realism measured directly against held-out
real rows. Without privacy, TabDDPM recovers most of what CTGAN lost (readmission
ROC-AUC 0.741 against 0.673, length-of-stay R² 0.135 against 0.017; both survive
correction; the largest model closes 74% and 82% of the gap to real data). At ε = 10 it
does not survive: private TabDDPM is indistinguishable from private CTGAN, at chance on
readmission and below the mean on length of stay, and a diagnostic run attributes the
collapse to the noise rather than the trainer. Model capacity interacts with privacy:
the size that is best without privacy is worst with it (significant on length of stay),
so realism and privacy do not commute. Pooling four private generators reduces
run-to-run spread tenfold only when the budget grows to ε ≈ 23, a patient-level
guarantee costs nothing measurable, and every release, private or not, sits at chance
against membership attacks. The study also found two defects in the programme's
frozen evaluation harness, per-split median imputation and thread-count-dependent
reproducibility; it corrects the first for its realism metrics and discloses both.

## 1. Motivation: the safeguards took the utility away

The first three pillars of this programme added one safeguard at a time to a
synthetic-EHR generator and measured what each cost. The measurements, hardened over
five seeds and patient-disjoint splits in the evidence repository, left two problems.

The first is that synthesis itself destroys most of the signal before any safeguard
is applied. Trained on real data, a random forest explains 18.6% of the variance in
ICU length of stay (R² 0.186 ± 0.001). Trained on non-private CTGAN output, it explains
0.9% (0.009 ± 0.083). For readmission the cost is smaller but real (ROC-AUC 0.778 →
0.690). On length of stay, then, the bottleneck is the generator, not the privacy
layer: differential privacy adds noise to a dataset that has already lost what it was
for.

The second is that differential privacy makes the generator unpredictable. Under DP
the run-to-run standard deviation of downstream utility grows 4.8-fold on readmission
and 17.9-fold on length of stay, and a single run cannot tell a data controller
whether the dataset about to be released is usable.

The realism pillar is therefore not "build a better generator", which is a crowded
and largely solved question. It is whether the utility the safeguards took away can be
recovered *inside* the constraints they impose, and whether the noise they introduced
can be tamed. That question did not exist until the safeguards did, which is why
realism is the last pillar and not the first.

## 2. Contributions

1. **A diffusion generator, from scratch, with a patient-level private trainer.** A
   compact TabDDPM (Gaussian diffusion for numeric columns, multinomial diffusion for
   categorical ones) and a DP-SGD trainer with true Poisson subsampling, per-example
   gradients, noise multiplicity, and an option to make the privacy unit the patient
   rather than the ICU stay. MIT-licensed, no third-party diffusion code.
2. **Realism measured directly.** Detection AUC, precision and recall, marginal and
   dependence errors, all against real rows the generator never saw, with a
   real-versus-real control that sets the ceiling. Stages 0 to 2 only ever measured
   realism through downstream utility.
3. **A controlled test of ensembling across DP runs**, at the per-run budget and at a
   fixed total budget, for both generator families.
4. **An empirical test of the ladder's ordering**: whether realism and privacy commute.
5. **Fairness at restored utility**, and the cost of a patient-level guarantee.
6. **A pre-registered analysis** ([`PREREGISTRATION.md`](PREREGISTRATION.md)), fixed before
   the main run, with each claim's decision rule and kill condition.

## 3. Method

### 3.1 The frozen harness

Everything that measured Stages 0 to 2 is vendored unchanged (`src/VENDORED.sha256`
records the hashes; `make verify-vendored` checks them): the loaders, the patient-
disjoint splits, the 30,000-row fitting sample, the downstream models (XGBoost for
readmission, a random forest for length of stay), the membership-inference attacks
and the fairness metrics. Seeds 42 to 46 are the evidence repository's, so every
split is identical and every comparison with the earlier stages is paired. The first
DP-CTGAN run of each seed uses Stage 1's recipe and seed exactly, which reproduces
the evidence repository's numbers and checks that nothing drifted.

### 3.2 TabDDPM

Numeric columns are mapped to a standard normal by a quantile transform and diffused
with Gaussian noise (Ho et al., 2020); the network predicts the noise. Discrete
columns are one-hot encoded and diffused with multinomial diffusion (Hoogeboom et al.,
2021) towards the uniform distribution; the network predicts the clean category, the
model posterior is computed from it in closed form, and the loss is the KL divergence
between true and model posteriors (which at the last step is the decoder likelihood).
One MLP denoiser sees both parts, with a sinusoidal timestep embedding added to a
linear projection of the input, as in Kotelnikov et al. (2023). The schedule is cosine
(Nichol & Dhariwal, 2021) with 1,000 steps; sampling uses an exponential moving average
of the weights.

Two implementation details turned out to matter, and both were caught by tests
before any real experiment ran. The moving average needs a warm-up: without one it
keeps (0.999)^t of the random initialisation, about 22% after 1,500 steps, which
visibly distorted sampled category shares in a test with a known answer. And on
Apple's MPS backend, Poisson-sized batches change tensor shapes every step and force
the graph to recompile; padding each gradient chunk to a fixed size, with the padding
assigned to a unit whose contribution is zeroed, brought a 100-epoch private fit back to
about four minutes (unpadded, the first pilot run had not finished after eight) without
changing the result: a test checks that the padding contributes exactly nothing.

### 3.3 The private trainer

DP-SGD (Abadi et al., 2016) with four deliberate choices.

* **True Poisson subsampling.** Each privacy unit enters a step independently with
  probability q, and the summed gradient is divided by the expected batch size, so
  every step is exactly the sampled Gaussian mechanism the Rényi accountant analyses
  (Mironov, Talwar & Zhang, 2019). Stages 1 and 2 used fixed-size batches accounted as
  Poisson, a standard approximation that they documented.
* **Per-example gradients** via `torch.func` (vmap over a gradient), tested against an
  explicit loop to 10⁻⁵.
* **Noise multiplicity** (Dockhorn et al., 2023): each unit's loss is averaged over
  four (timestep, noise) draws before its gradient is clipped. The averaging happens
  inside one unit's contribution, so it reduces gradient variance at no privacy cost.
* **Stage 1's noise convention.** Noise is calibrated to sensitivity 2C and to ε = 10 by
  the same accountant, so "ε = 10" means the same noise level on every rung and the
  generators are compared at an identical privacy cost. Against add/remove adjacency,
  the adjacency for which the accountant is exact under Poisson sampling, the same
  runs satisfy a much smaller ε (4.38 for a DP-TabDDPM run; 3.57 for a DP-CTGAN run,
  under its fixed-batch approximation), which is also reported.

**Patient-level privacy.** With `unit = "patient"`, a patient's stays are sampled
together and their gradients summed before clipping, so the guarantee covers a
patient's whole contribution. In the 30,000-row fitting samples, 28 to 30% of rows
come from patients with more than one stay, up to 26 stays for one patient.

What the guarantee does not cover, as in Stage 1: the quantile transform and each
discrete column's category list are fitted on the real training rows (DP-CTGAN's
Gaussian-mixture normalisation has the same status), and the hyperparameters were
chosen on private data (Section 3.6).

### 3.4 Pooled releases

A pooled release takes n/K rows from each of K independently trained DP generators
fitted on the same rows. Because all K runs touch the same data, their Rényi curves
add; no parallel-composition discount is claimed (`src/budget.py`). Two regimes:

* **full budget:** each run at ε = 10 on its own, so the pool spends the composed
  budget. Rényi composition is sub-linear: four runs compose to ε = 23.5 for
  DP-TabDDPM and 22.8 for DP-CTGAN, not 40;
* **split budget:** a fixed total of ε = 10, each run calibrated so that the four
  compose to exactly that.

### 3.5 Fair resampling

Stage 2's fair sampling conditioned the CTGAN generator on a sensitive attribute.
Diffusion models are unconditional here, so fair sampling is done by drawing a large
pool from the private generator and resampling it to the fair target, uniform over
groups with at least 2% share (Stage 2's target). Weights come from raking
(iterative proportional fitting), which balances several sensitive marginals jointly,
and the release is drawn by exact stratified sampling within the raked cells. The
pool, its shares and the target are all functions of the private generator's output,
so the step costs no additional ε. The same procedure is applied to DP-CTGAN so that
the two families are compared like for like.

### 3.6 The pilot

The only hyperparameter search ran once, before the main experiment, on seed 42's
training rows with a patient-disjoint 15% validation split; no test set was touched.

| Setting | LoS R² | LoS detection | Readm. ROC-AUC | Readm. detection |
|---|---|---|---|---|
| real training rows (validation ceiling) | 0.178 | – | 0.772 | – |
| TabDDPM, 5k steps | 0.119 | 0.713 | 0.693 | 0.997 |
| TabDDPM, 10k steps | 0.123 | 0.675 | 0.725 | 0.983 |
| **TabDDPM, 20k steps** | **0.131** | 0.632 | **0.733** | 0.958 |
| DP-TabDDPM, batch 1,024, 100 epochs | −3.877 | 1.000 | 0.403 | 1.000 |
| DP-TabDDPM, batch 4,096, 100 epochs | −0.816 | 1.000 | 0.473 | 1.000 |
| **DP-TabDDPM, batch 2,048, 200 epochs** | **0.026** | 0.999 | **0.488** | 1.000 |

The selection rule (best mean rank across the two tasks) was committed before the
readmission DP settings had finished, and picked the bold rows. After the length-of-stay
DP results, a diagnostic run of the same private trainer (`src/diagnose_dp.py`, first
DP setting) located the collapse:

| Private trainer with | LoS R² | Detection | Precision |
|---|---|---|---|
| no clipping, no noise | 0.065 | 0.885 | 0.439 |
| clipping at C = 1, no noise | 0.109 | 0.714 | 0.852 |
| clipping and noise (ε = 10) | −3.877 | 1.000 | 0.000 |

The trainer and the clipping are sound; the noise is what destroys the model. (Clipping
even helps slightly, acting as gradient normalisation.)

### 3.7 Analysis

As pre-registered: paired t-tests across the five seeds for each primary comparison,
Holm-adjusted across the family; every effect is compared with its family's noise
floor, the mean absolute difference between independent replicates of the same
recipe; spread is compared with the Brown-Forsythe test; the minimum detectable paired
effect uses the evidence repository's definition, (t₀.₉₇₅ + t₀.₈₀) · sd(d) / √n.

## 4. Threats to validity, stated in advance

* **Five seeds.** Wide intervals; a null is reported as unresolved, not as absent.
* **One dataset.** MIMIC-IV only.
* **Tuning on private data.** Disclosed in Section 3.6.
* **The comparison is at a fixed generator budget, not a fixed compute budget.**
  DP-TabDDPM and DP-CTGAN are each given their own reasonable training length.
* **The noise convention is conservative.** Both private generators spend the same
  noise; under the more common add/remove convention both would be reported at a
  smaller ε.

## 5. Results

All values are means with 95% t-intervals over seeds 42 to 46. The full tables are in
[`results/results_table.md`](../results/results_table.md).

### 5.1 Hypothesis verdicts

Fourteen comparisons fixed in advance, paired t-tests across five seeds, Holm-adjusted
across all fourteen. Δ is the first arm minus the second, in ROC-AUC (readmission) or R²
(length of stay). The noise floor is the mean absolute difference between independent
replicates of the private recipe involved; it is defined only for private recipes.

| Hypothesis | Comparison | Readmission Δ (p, Holm p) | Length of stay Δ (p, Holm p) | Verdict |
|---|---|---|---|---|
| H1 | TabDDPM − CTGAN | +0.068 (0.0002, **0.003**) | +0.118 (0.0004, **0.005**) | **supported on both tasks** |
| H2 | DP-TabDDPM − DP-CTGAN | +0.024 (0.43, 1.0) | −0.501 (0.28, 1.0) | not supported: within noise |
| H3b | split-budget pool − single run, TabDDPM | −0.001 (0.96, 1.0) | −4.049 (0.031, 0.31) | pool no better; worse on LoS before correction |
| H3b | split-budget pool − single run, CTGAN | −0.001 (0.96, 1.0) | −0.417 (0.25, 1.0) | pool no better |
| H4 | (L − S) without DP − (L − S) with DP | +0.144 (0.007, 0.078) | +10.250 (0.002, **0.023**) | **supported on LoS**; suggestive on readmission |
| H5 | fair resampling − none, TabDDPM | +0.017 (0.32, 1.0) | +0.469 (0.16, 1.0) | no measurable cost |
| H6 | patient-level − stay-level DP | +0.002 (0.96, 1.0) | +0.374 (0.39, 1.0) | no measurable cost |

Noise floors (mean absolute difference between replicates): readmission 0.049 (DP-CTGAN)
and 0.059 (DP-TabDDPM) ROC-AUC; length of stay 0.942 and 0.712 R². H3a (spread) is a
Brown-Forsythe test, reported in below.

**H1 (realism without privacy): supported.** Non-private TabDDPM beats CTGAN on both
tasks: readmission ROC-AUC 0.741 ± 0.007 against 0.673 ± 0.019, length-of-stay R²
0.135 ± 0.005 against 0.017 ± 0.032. It closes 65% and 70% of the gap between CTGAN and
real data; the 4 × 512 variant (0.751 and 0.155) closes 74% and 82%. Realism measured
directly agrees on length of stay (detection AUC 0.639 against 1.000; coverage 0.940
against 0.605, real data 0.965) and partly on readmission, where the medium model is
still easily detected (0.950) and only the large model approaches the real data
(detection 0.767, coverage 0.896 against 0.966).

**H2 (realism under privacy): not supported.** At ε = 10 the difference between the
private generators is within noise on both tasks (readmission 0.516 against 0.492;
length of stay −0.840 against −0.339). Both are at chance on readmission, both make
length-of-stay predictions worse than the mean, and every private release is trivially
detectable (detection AUC 1.000, coverage 0.000 to 0.009). The better generator does
not survive privacy. Section 3.6's diagnostic places the cause in the noise.

**H3a (pooling at full budget reduces spread): direction supported, not resolved.**
For DP-TabDDPM on length of stay, the standard deviation falls from 0.525 (twenty single
runs) to 0.050 (five pools of four) and the mean rises from −0.840 to −0.080, at a
composed ε of 23.5. The Brown-Forsythe test does not resolve the reduction with five
pools (p = 0.32). On readmission, where single runs are already at chance, pooling
changes nothing. DP-CTGAN behaves the same way (length of stay: 1.383 to 0.247,
p = 0.42).

**H3b (pooling at split budget does not beat one run): supported, and worse.** With a
fixed ε = 10 shared by four runs, each run is much noisier (noise multiplier 4.28
against 2.25), and on length of stay the pool is worse than one run (−4.889 against
−0.840; p = 0.031 before correction, 0.31 after). For DP-CTGAN the split pool is no
better either (−0.756 against −0.339, within noise). Predictability is bought with
privacy budget, not created.

**H4 (realism and privacy do not commute): supported on length of stay, suggestive on
readmission.** Without privacy, utility rises with denoiser size on both tasks (S, M, L:
0.641, 0.741, 0.751 and 0.128, 0.135, 0.155). Under privacy the order reverses (0.538,
0.516, 0.504 and 0.028, −0.840, −10.195). The difference-in-differences is large on
length of stay and survives Holm correction (p = 0.002, Holm p = 0.023); on
readmission it is significant only before correction (p = 0.007, Holm p = 0.078).

**H5 (fairness becomes measurable at restored utility): not supported under privacy;
supported without it.** Every DP-TabDDPM classifier predicts a single class in all 35
subgroup cells, so the threshold-based fairness metrics are vacuous for every private
release. The explanation is the label rate (Section 5.3). The large non-private
generator, whose label rate is close to the real one, gives a classifier whose
disparities track the real-data model (demographic-parity gap 0.302 ± 0.030 against
0.309 ± 0.032; AUC gap 0.070 against 0.075). Fair resampling costs nothing measurable
on either task, as in Stage 2.

**H6 (patient-level privacy costs nothing): supported.** Making the patient the
privacy unit changes nothing beyond the noise (readmission +0.002, p = 0.96; length of
stay +0.374, p = 0.39, in the patient-level arm's favour). The per-stay guarantee that
the risk register carried since Stage 1 can be replaced by a per-patient one for free at
this budget.

### 5.2 Privacy

Every synthetic release sits at chance against both frozen membership attacks (AUC
0.497 to 0.503) and at the chance level of the low-false-positive regime (true-positive
rate at 1% false positives 0.009 to 0.011). The same attacks score 1.000 when the
"release" is the real training data, so they are capable of detecting a leak; they
simply find none, with or without privacy. Stage 3's release gate, re-applied to every
arm ([`results/release_check.md`](../results/release_check.md)), therefore passes every
release on its empirical criterion and separates them only by the formal one: Stage 1's
argument that only the formal bound can certify a release holds for the new generator.

### 5.3 The label rate

The downstream classifier weights its classes by the release's own positive rate. A
release with the wrong rate gives a classifier that, at the 0.5 threshold, predicts one
class: the medium non-private TabDDPM generates 0.375 ± 0.104 positives against a real
0.139 and is degenerate in 19 of 35 subgroup cells; the large one generates 0.155 ±
0.011 and is degenerate in none. ROC-AUC is blind to this, which is why the medium
model's good AUC coexists with useless thresholded predictions. Label-rate fidelity is
reported for every arm ([`src/label_rate.py`](../src/label_rate.py)).

### 5.4 Reproduction, and three measurement problems

Re-running Stage 0 and Stage 1's recipes with the evidence repository's seeds did not
reproduce its numbers bit for bit (readmission ROC-AUC differs by up to 0.043 non-private
and 0.061 private), because the runs here use two CPU threads where the evidence run
used twelve: summation order changes and adversarial training amplifies the difference.
Every comparison in this paper uses this repository's own runs. Two further problems
were found in the measurement itself and corrected before the analysis: the frozen
harness imputes each split with its own median, which let a classifier tell real
training from real test rows (detection AUC 0.956 in two readmission seeds, fixed by
imputing the test split with training medians for the realism metrics); and improved
precision/recall rewards over-dispersed private samples with recall near 1, so density
and coverage were added. Both additions are post hoc, are labelled as such, and change
no pre-registered utility or privacy test.

## 6. Discussion

**What the pillar shows.** Realism can be recovered, and substantially: the generator,
not the data, was the bottleneck that Stage 0 exposed on length of stay. But at the
programme's release budget, privacy destroys what the better generator recovers, and
the generator that is most realistic without privacy is the least robust with it. For a
data controller, that is the practical conclusion: a generator should be chosen and
tuned under the privacy constraint it will be released with, and at ε = 10 with this
family of private trainers, the honest release is one whose low utility is disclosed,
not one that is presented as a realistic stand-in for the data.

**What it does not show.** It does not show that no private generator could do better.
Marginal-based private synthesisers, private pre-training on public data, or a larger
budget are all untested here. The ladder tested whether the route it chose recovers
realism inside its own safeguards, and the answer is that it recovers it outside them.

**What carries forward.** The measurement discipline: a pre-registered design, a noise
floor measured for every new recipe, controls that can fail (the real-data attack
control, the real-versus-real realism control), and three problems in the measurement
itself found because those controls were in place and acted on when they misbehaved.

## 7. Relation to the ladder

| Stage | Question | Pillar 4's contribution |
|---|---|---|
| 0, synthesis | How much survives synthesis? | Much more, with a better generator: 74 to 82% of the gap closed |
| 1, privacy | What does privacy cost? | Everything the better generator recovered; patient-level privacy costs nothing more |
| 2, fairness | Can representation be repaired? | Fairness becomes measurable only at restored utility, which privacy prevents |
| 3, compliance | Can evidence be generated? | Re-applied, the gate's empirical check passes every release, private or not; only ε separates them |
| evidence | Which claims survive noise? | Noise floors for the new recipe; two harness defects disclosed |

## References

Abadi, M., et al. (2016). Deep learning with differential privacy. *CCS*.
Carlini, N., et al. (2022). Membership inference attacks from first principles. *IEEE S&P*.
Deming, W. E., & Stephan, F. F. (1940). On a least squares adjustment of a sampled frequency table when the expected marginal totals are known. *Annals of Mathematical Statistics*, 11(4).
Dockhorn, T., Cao, T., Vahdat, A., & Kreis, K. (2023). Differentially private diffusion models. *TMLR*.
Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., & Welling, M. (2021). Argmax flows and multinomial diffusion. *NeurIPS*.
Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023). TabDDPM: Modelling tabular data with diffusion models. *ICML*.
Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019). Improved precision and recall metric for assessing generative models. *NeurIPS*.
Lopez-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests. *ICLR*.
Mironov, I., Talwar, K., & Zhang, L. (2019). Rényi differential privacy of the sampled Gaussian mechanism. *arXiv:1908.10530*.
Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML*.
Papernot, N., & Steinke, T. (2022). Hyperparameter tuning with Rényi differential privacy. *ICLR*.
Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *NeurIPS*.
Johnson, A., et al. (2024). MIMIC-IV (version 3.1). *PhysioNet*.
