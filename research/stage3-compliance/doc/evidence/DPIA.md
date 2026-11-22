# Data Protection Impact Assessment: DP-Fair Synthetic EHR Release (Stage 3)

*Compiled from pipeline artefacts on 2026-07-12. The structure follows the UK
ICO's DPIA template, Steps 1 to 7. Every figure below is injected from the
generation manifests and evaluation metrics, and none is entered by hand. This
document is a research artefact demonstrating a compliance methodology. It is not
legal advice, and it does not by itself authorise any disclosure.*

## Step 1. Describe the processing

**Nature.** Generation of synthetic electronic-health-record feature matrices by
Fair-DP-CTGAN (fair conditional generation on a DP-SGD critic; RDP-accounted), trained on derived, de-identified MIMIC-IV v3.1 cohorts accessed
under the PhysioNet Credentialed Health Data Use Agreement, within a single
credentialed research environment. The release candidate is the
`fairsample` arm: 79,693 synthetic ICU-stay rows for the length-of-stay
task, 69,699 rows for the readmission task, and one synthetic daily count
series for bed demand. No real patient-level record is stored in, or emitted by,
the release artefacts.

**Scope.** Training touches real data at only two points: a critic trained under
DP-SGD (100 epochs on a 30,000-row fitting subsample, at a
noise multiplier of 1.01), and a single Gaussian-mechanism release of
discrete-category counts. Both are composed in a Rényi differential privacy
accountant. The sensitive attributes processed are: gender; insurance (readmission), gender (LOS).

**Context and purpose.** An independent research programme evaluating whether synthetic
EHR can support hospital predictive modelling for length of stay, 30-day readmission
and bed demand, while bounding re-identification risk, mitigating subgroup bias, and
remaining auditable.

## Step 2. Consultation

Internally, the author reviews this evidence before any promotion. Externally, a
practitioner and governance review of this evidence package is a standing open human
gate, recorded in the release decision; the framework blocks external release until
it is signed. Consultation with data subjects is not feasible, since the underlying
database is retrospective and de-identified. That limitation is inherited from, and
mitigated by, the PhysioNet access model.

## Step 3. Necessity and proportionality

The purpose, which is shareable analytic data that does not expose patients, cannot
be met by the real data under the data use agreement, nor by ad-hoc anonymisation,
which offers no composable guarantee and has a known history of failure on
high-dimensional clinical data. Differential privacy is the least intrusive effective
mechanism available, because it provides a worst-case, attacker-agnostic bound. The
release is made at a budget of 9.99 for readmission and 9.99 for
length of stay, at a delta of 1e-5, which is below the reciprocal of the cohort
size in both cases, with per-day accounting at 1e-4 for the count series.
Data are minimised to model-ready feature matrices, and no identifiers, free text or
dates enter the generator.

## Step 4. Identify and assess the risks

| Ref | Risk | Likelihood | Severity | Assessment evidence |
|---|---|---|---|---|
| R1 | Membership inference from the release | Low | High | Formal budget, plus black-box attacks at chance: length of stay 0.497 and 0.503, readmission 0.499 and 0.501 (0.5 indicates no signal) |
| R2 | Memorised near-copies of training rows | Low | High | Near-duplicate share at a distance of 0.1: 0.000 for length of stay, 0.000 for readmission |
| R3 | Residual identifiability outside the formal bound, from preprocessing statistics and row-level adjacency | Medium | Medium | Risk-register items RR-1 to RR-3, openly documented, with refinements scheduled |
| R4 | Representation harm: the release misrepresents subgroups | Medium | Medium | Subgroup shares are unstable under differential privacy. Medicare share: 0.542 in the real cohort, 0.421 in the unmitigated arm, 0.368 in the released arm. See the risk register and the Stage-2 paper |
| R5 | Misuse: models trained on the release are treated as clinically valid | Medium | High | Utility is disclosed honestly, including negative results (readmission AUC 0.451 against a real-data ceiling of 0.770; length-of-stay R² -0.192 against 0.186). The model card forbids clinical use |
| R6 | Contractual breach of data-use-agreement limits on redistributing derived data | n/a | High | The `dua` human gate is open, and external release is held |

## Step 5. Measures to reduce risk

A DP-SGD critic with an accounted frequency release addresses R1 and R2. The
residual channels behind R3 are documented rather than concealed, with a DP quantile
normalisation and patient-level adjacency scheduled as refinements. Fair conditional
sampling, at no additional privacy cost, addresses R4, though Section 4 of the
Stage-2 paper records that its effect is to stabilise representation rather than to
repair a systematic bias. Mandatory joint disclosure of utility alongside fairness,
and an explicit prohibition of clinical use in the model card, address R5. A two-tier
release decision, whose human gates code cannot close, addresses R6.

## Step 6. Sign-off and outcomes

The technical gates are recorded in `results/release_decision.json`, which is machine
generated and reproducible. Internal promotion to the Stage-3 dataset folder is
decided by those gates alone. External release is held until the `dua` and
`stakeholder` human gates are signed. Residual risk after mitigation is assessed as
medium to low, dominated by the R3 items, and is accepted for internal research use
only.

## Step 7. Integration and review

The compiler regenerates this assessment from the pipeline's artefacts on every run,
and it is version-controlled alongside the code that produced the data. Review is
triggered by any upstream regeneration, any change to the privacy budget, the
completion of the realism pillar, and the application of this framework to a second
dataset, which will serve as its reusability test.
