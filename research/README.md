# The research behind SynthEHR

Every claim the [tool](../README.md) makes was first measured here, on MIMIC-IV, one
safeguard at a time. This folder is the complete record: code exactly as it ran,
configurations, committed results (aggregates only), and a write-up per stage.

## Reading order

| Step | Folder | Question | Write-up |
|---|---|---|---|
| Foundations | [`foundations/`](foundations) | What can real data do? Three published prediction models replicated on MIMIC-IV set the ceilings every synthetic result is measured against. | [length of stay](foundations/length-of-stay/README.md) · [readmission](foundations/readmission/README.md) · [bed demand](foundations/bed-demand/README.md) |
| Stage 0 | [`stage0-synthesis/`](stage0-synthesis) | How much predictive signal survives synthesis with a from-scratch CTGAN? | [README](stage0-synthesis/README.md) · [paper](stage0-synthesis/doc/research-paper.md) |
| Stage 1 | [`stage1-privacy/`](stage1-privacy) | What does a formal differential-privacy guarantee cost, and do attacks notice it? | [README](stage1-privacy/README.md) · [paper](stage1-privacy/doc/research-paper.md) |
| Stage 2 | [`stage2-fairness/`](stage2-fairness) | Can subgroup representation be repaired without spending more privacy? | [README](stage2-fairness/README.md) · [paper](stage2-fairness/doc/research-paper.md) |
| Stage 3 | [`stage3-compliance/`](stage3-compliance) | Can release evidence be generated rather than written, and can a gate refuse? | [README](stage3-compliance/README.md) · [paper](stage3-compliance/doc/research-paper.md) |
| Evidence | [`evidence/`](evidence) | Which of those findings survive five seeds and a measured noise floor? | [README](evidence/README.md) |
| Stage 4 | [`stage4-realism/`](stage4-realism) | Can realism be recovered inside the safeguards? Pre-registered. | [README](stage4-realism/README.md) · [paper](stage4-realism/doc/research-paper.md) · [pre-registration](stage4-realism/doc/PREREGISTRATION.md) |

The evidence layer comes before Stage 4 on purpose: it measured the run-to-run noise
of private generators, retracted two of the earlier stages' single-run claims, and set
the rule that every later claim is tested against a noise floor.

## How the code is organised

* [`common/`](common) holds the modules every stage shares (the CTGAN, the data
  loaders, the privacy accountant, the attacks). They are byte-identical to the
  versions that produced the published numbers; [`FROZEN.sha256`](common/FROZEN.sha256)
  records their hashes and CI checks them. Each stage's `Makefile` puts this folder on
  `PYTHONPATH`. Their docstrings predate this repository and still describe the
  original multi-repository layout.
* Each stage folder holds its own scripts, `config.yaml`, `results/` and write-up.
* The maintained, documented, tested implementation is the [`synthehr`](../src/synthehr)
  package. Its tests check that it reproduces these modules bit for bit.

## Reproducing

You need credentialed access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
(PhysioNet). Then:

```bash
pip install -r research/requirements.txt

# 1. build the real-data inputs (set mimic_root / mimic_dir in each config.yaml first)
make -C research/foundations/length-of-stay extract train eval
make -C research/foundations/readmission run
make -C research/foundations/bed-demand all

# 2. any stage, from its own folder
make -C research/stage0-synthesis all
make -C research/evidence run noise aggregate
make -C research/stage4-realism run realism aggregate plots
```

Real-data inputs are written inside each foundation's `data/` folder; synthetic
datasets go to `../synthehr-output/` next to the repository. Neither is ever
committed. Long runs are resumable: every seed or arm writes its own result file and
is skipped if present.

## What was found wrong, and where it is reported

This programme reports its own errors where a reader will see them:

| Issue | Where it is reported |
|---|---|
| A row-level split leaked patients into the test set (43% of readmission test rows) | [evidence](evidence/README.md), fixed before any reported number |
| Two single-run headline claims did not survive five seeds | [evidence](evidence/README.md#53-what-survived-and-what-did-not), retracted |
| The noise control re-seeded the fitting sample as well as the model | [evidence](evidence/README.md#3-method) |
| Each split imputed with its own median | [evidence](evidence/README.md#6-limitations), [Stage 4](stage4-realism/README.md#76-three-measurement-problems-found-and-fixed) |
| A second, uncharged noisy count release in the fair DP-CTGAN (ε 9.9922, not 9.9902) | [Stage 2](stage2-fairness/README.md#8-limitations) |
| A seed does not reproduce a GAN run across CPU thread counts | [Stage 4](stage4-realism/README.md#76-three-measurement-problems-found-and-fixed) |
| Small cells in a per-unit table (units with fewer than 10 stays) | [length-of-stay foundation](foundations/length-of-stay/README.md), now suppressed |

## Data use

MIMIC-IV is used under the PhysioNet Credentialed Health Data Use Agreement. No real or
synthetic patient-level data, and no model trained on it, is in this repository; only
aggregate metrics and figures, with cells under 10 patients suppressed.
