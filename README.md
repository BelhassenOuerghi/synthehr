# Realism under privacy: Pillar 4 of a synthetic-EHR programme

*An EngD research study. Belhassen Ouerghi, University of Hertfordshire.*

*Programme: [Stage 0, CTGAN baseline](https://github.com/BelhassenOuerghi/mimic-synthetic-ctgan) · [Stage 1, privacy](https://github.com/BelhassenOuerghi/mimic-synthetic-dp-ctgan) · [Stage 2, fairness](https://github.com/BelhassenOuerghi/mimic-synthetic-fair-dp-ctgan) · [Stage 3, compliance](https://github.com/BelhassenOuerghi/mimic-synthetic-ethic-fair-dp-ctgan) · [Stage 4, realism](https://github.com/BelhassenOuerghi/mimic-synthetic-realism) · [Evidence: five seeds, noise floor](https://github.com/BelhassenOuerghi/mimic-synthetic-evidence)*

## 1. Overview

<!-- OVERVIEW: headline findings, written from the results -->

## 2. The programme in one table

| Stage | Repository | Question | What it found |
|---|---|---|---|
| 0 | [`mimic-synthetic-ctgan`](https://github.com/BelhassenOuerghi/mimic-synthetic-ctgan) | How much predictive signal survives synthesis? | Less than hoped: readmission ROC-AUC 0.778 → 0.690; length-of-stay R² 0.186 → 0.009 |
| 1 | [`mimic-synthetic-dp-ctgan`](https://github.com/BelhassenOuerghi/mimic-synthetic-dp-ctgan) | What does a formal privacy guarantee cost? | Differential privacy makes the generator unpredictable; attacks sit at chance even without privacy, so only the formal bound can certify a release |
| 2 | [`mimic-synthetic-fair-dp-ctgan`](https://github.com/BelhassenOuerghi/mimic-synthetic-fair-dp-ctgan) | Can subgroup representation be repaired for free? | Fair sampling costs no extra ε; its utility cost is below the noise floor; DP makes representation unstable rather than biased in a fixed direction |
| 3 | [`mimic-synthetic-ethic-fair-dp-ctgan`](https://github.com/BelhassenOuerghi/mimic-synthetic-ethic-fair-dp-ctgan) | Can compliance evidence be generated rather than written? | An evidence compiler and a release gate that can fail; a compliant release can be useless |
| 4 | this repository | Can realism be recovered inside the safeguards? | <!-- ROW4 --> |
| – | [`mimic-synthetic-evidence`](https://github.com/BelhassenOuerghi/mimic-synthetic-evidence) | Which findings survive seed noise? | A measured noise floor; two claims retracted |

## 3. Why realism is the last pillar

The safeguards of Stages 1 to 3 define what may be released; realism is optimised
inside that region, not before it. Starting from the strongest generator available
would have made this pillar unfalsifiable, because there would have been nothing
to recover. And the evidence repository showed two different problems for realism
to attack. On length of stay, synthesis itself destroyed the signal before privacy
was applied, so the bottleneck is the generator. On readmission, the privacy layer
did most of the damage. A realism intervention that only touched one of the two
could not have fixed both.

## 4. What I built

* **TabDDPM from scratch** ([`src/tabddpm.py`](src/tabddpm.py)): Gaussian diffusion for
  numeric columns, multinomial diffusion for categorical ones, an MLP denoiser with a
  timestep embedding, a cosine schedule and a weight average for sampling
  (Kotelnikov et al., 2023). MIT-licensed, no third-party diffusion code.
* **A private trainer for it**: DP-SGD with true Poisson subsampling, per-example
  gradients (`torch.func`), noise multiplicity, and a switch that makes the privacy
  unit the **patient** rather than the ICU stay. The noise is calibrated in exactly
  Stage 1's convention, so ε = 10 means the same noise level on every rung.
* **Realism metrics** ([`src/fidelity.py`](src/fidelity.py)): detection AUC, precision and
  recall, marginal and dependence errors, against real rows no generator saw, with a
  real-versus-real ceiling; and the true-positive rate at 1% and 0.1% false positives
  for the membership attacks.
* **Pooled releases with exact composition** ([`src/budget.py`](src/budget.py)) and
  **fair resampling** for unconditional generators ([`src/fair_resample.py`](src/fair_resample.py)).
* **Stage 3's release gate, re-applied** to every candidate ([`src/release_check.py`](src/release_check.py)).

## 5. Data

The derived, model-ready feature matrices of the two tabular replication studies
([length of stay](https://github.com/BelhassenOuerghi/Length-of-Stay-Prediction-Replica),
[readmission](https://github.com/BelhassenOuerghi/readmission-prediction-replica)) on MIMIC-IV
v3.1, with the evidence repository's patient-disjoint splits and seeds. Each generator
is fitted on a 30,000-row sample of the training split and releases as many rows as
the training split holds (69,699 for readmission and 79,693 for length of stay at seed
42). Bed demand needs no realism recovery: its DP generator already forecasts as well as
the real series (evidence repository, MAE 1.460 against 1.475), so it is not re-run here.

## 6. Method

<!-- METHOD SUMMARY -->

The design, the hypotheses and the decision rules were fixed in
[`doc/PREREGISTRATION.md`](doc/PREREGISTRATION.md) before the main run, and the full
method is in [`doc/research-paper.md`](doc/research-paper.md).

## 7. Results

<!-- RESULTS -->

## 8. Reproducibility

```bash
PY=python3   # any Python with requirements.txt installed

make test             # unit tests on fabricated data, no MIMIC needed
make verify-vendored  # the frozen Stage 0-2 harness is byte-identical
make pilot            # the one hyperparameter search (seed 42 training rows only)
make run              # every arm and seed; resumable, one subprocess per arm
make aggregate        # intervals and the pre-registered tests -> results/
make plots            # figures -> results/plots/
```

Paths in [`config.yaml`](config.yaml) are relative to this repository; set
`ENGD_REPOS_ROOT` and `ENGD_SYNTH_ROOT` to point elsewhere. Synthetic datasets are
written outside the repository. Per-arm, per-seed results are committed under
`results/arms/` so every number in this README can be traced to a file.

## 9. Limitations

<!-- LIMITATIONS -->

## 10. Data use and ethics

PhysioNet Credentialed Health Data Use Agreement, MIMIC-IV v3.1. No real or synthetic
patient-level data is committed; only aggregate metrics are. Synthetic data derived
from MIMIC is treated as derived data under the agreement and is not redistributed.
Research artefacts, not medical devices.

### References

See [`doc/research-paper.md`](doc/research-paper.md#references).
