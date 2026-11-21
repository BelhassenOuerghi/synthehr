# Realism Under Privacy: the Fourth Pillar of a Synthetic-EHR Programme

*Belhassen Ouerghi, EngD, University of Hertfordshire.*

*Draft. Sections 1 to 4 were written before the main experiment's results existed;
Sections 5 onwards report them.*

## Abstract

<!-- ABSTRACT: written last, from Section 5 -->

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
  runs satisfy a much smaller ε (about 3.5 to 4.4 here), which is also reported.

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
  budget. Rényi composition is sub-linear: four runs compose to about ε = 23 for
  DP-TabDDPM, not 40;
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

<!-- PILOT TABLE -->

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

<!-- RESULTS -->

## 6. Discussion

<!-- DISCUSSION -->

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
