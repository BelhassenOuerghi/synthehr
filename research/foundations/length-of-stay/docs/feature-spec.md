# Feature Specification: MIMIC-IV v3.1

This document defines the predictor set used to model ICU Length of Stay (LOS)
and maps each feature to its MIMIC-IV v3.1 source table, itemid(s), and
extraction logic. It reflects what `src/extract.py` actually computes.

> **Adaptation note.** This project was scaffolded against MIMIC-III (an
> earlier draft of this file listed 32 MIMIC-III predictors keyed on shifted
> DOB and CareVue/MetaVision itemids). It has been ported to MIMIC-IV v3.1,
> whose schema is lowercase, MetaVision-only, and ships a precomputed `los`.
> The feature set below is the MIMIC-IV realisation.

All time-varying physiology is aggregated over the **first 24 hours** after
ICU admission (`intime .. intime + 24h`), a window that is known near
admission and avoids using discharge-adjacent information.

## Demographics & admission context (5)

| Feature | Source | Logic |
|---|---|---|
| `age` | `hosp/patients.anchor_age` | Age in years at ICU admission (numeric) |
| `gender` | `hosp/patients.gender` | Categorical (M/F), one-hot encoded |
| `admission_type` | `hosp/admissions.admission_type` | Categorical, one-hot encoded |
| `unitcode` | `icu/icustays.first_careunit` | ICU unit; categorical + SLOS grouping key |
| `pre_icu_los_days` | `icu/icustays.intime` − `hosp/admissions.admittime` | Hospital days elapsed before ICU admission (clamped ≥ 0) |

## First-24h vital signs: `icu/chartevents` (15)

For each concept the mean, min, and max over the first-24h window are used
(`*_mean`, `*_min`, `*_max`). Values outside physiologic bounds are dropped in
SQL before aggregation so a single erroneous reading cannot dominate a min/max.

| Concept | itemid(s) | d_items label | bounds |
|---|---|---|---|
| `hr` (heart rate) | 220045 | Heart Rate | 10 to 300 bpm |
| `rr` (respiratory rate) | 220210 | Respiratory Rate | 1 to 80 insp/min |
| `spo2` (oxygen saturation) | 220277 | O2 saturation pulseoxymetry | 10 to 100 % |
| `temp` (temperature, °C) | 223762; 223761 | Temperature Celsius; Temperature Fahrenheit ( to  °C) | 25 to 45 °C |
| `map` (mean arterial pressure) | 220052, 220181, 225312 | Arterial / Non-Invasive / ART BP mean | 10 to 250 mmHg |

## First-24h laboratory values: `hosp/labevents` (9)

Mean over the first-24h window. `labevents` has no `stay_id`, so a draw is
attributed to a stay via `hadm_id` **and** the charttime falling inside that
stay's first-24h window.

| Feature | itemid(s) | d_labitems label | bounds |
|---|---|---|---|
| `creatinine` | 50912 | Creatinine | 0 to 50 mg/dL |
| `wbc` | 51301, 51300 | White Blood Cells / WBC Count | 0 to 200 K/µL |
| `glucose` | 50931 | Glucose | 0 to 2000 mg/dL |
| `sodium` | 50983 | Sodium | 80 to 200 mEq/L |
| `potassium` | 50971 | Potassium | 0 to 15 mEq/L |
| `bicarbonate` | 50882 | Bicarbonate | 0 to 60 mEq/L |
| `bun` | 51006 | Urea Nitrogen | 0 to 300 mg/dL |
| `hematocrit` | 51221 | Hematocrit | 0 to 100 % |
| `platelets` | 51265 | Platelet Count | 0 to 2000 K/µL |

**Total: 29 predictors** (2 numeric demographic/context + 15 vital aggregates
+ 9 lab means + 3 categorical, the last one-hot encoded at train time).

## Rationale vs upstream

The upstream Brazilian SLOS model used a bespoke registry predictor list
(demographics, admission source, first-hour vitals/labs, treatment and
comorbidity flags derived from a curated ICD mapping). MIMIC-IV does not carry
those registry fields natively, and its diagnoses mix ICD-9/ICD-10, so instead
of forcing a brittle code-mapping I selected a compact, defensible set that is
**analogous in spirit**: routinely-charted first-day physiology plus
demographics and admission context. This keeps the modelling target and
architecture identical to upstream while being faithful to what MIMIC-IV can
supply reliably.

## Missingness

Physiology missingness under the 24h window is low (vitals < 3% null, labs
≈ 3 to 4% null) because these are near-universally charted on ICU day 1. Missing
values are median-imputed (numeric) or most-frequent-imputed (categorical)
**inside** the model pipeline, so imputation statistics are learned within each
CV fold and never leak from test to train.
