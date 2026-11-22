# Replication Results: MIMIC-IV v3.1 vs Upstream

Held-out test set: 14,037 ICU stays (9,786 patients), patient-level split (seed 42).

| Metric | This replication (MIMIC-IV) | Upstream benchmark |
|---|---:|---:|
| RMSE (days) | 3.708 | 4.02 |
| MAE (days) | 2.343 | 2.67 |
| R^2 (patient-level) | 0.171 | 0.17 |
| R^2 (unit-level) | 0.994 | 0.91 |
| Overall SLOS | 1.001 | 1.00 (by construction) |

## Per-ICU-unit SLOS

| ICU unit (first_careunit) | Admissions | Observed LOS (d) | Predicted LOS (d) | SLOS |
|---|---:|---:|---:|---:|
| Medical Intensive Care Unit (MICU) | 2,995 | 10888 | 11051 | 0.985 |
| Medical/Surgical Intensive Care Unit (MICU/SICU) | 2,301 | 6894 | 7515 | 0.917 |
| Cardiac Vascular Intensive Care Unit (CVICU) | 2,295 | 7308 | 6794 | 1.076 |
| Surgical Intensive Care Unit (SICU) | 1,910 | 6943 | 6693 | 1.037 |
| Coronary Care Unit (CCU) | 1,603 | 4952 | 5222 | 0.948 |
| Trauma SICU (TSICU) | 1,487 | 5341 | 5185 | 1.030 |
| Neuro Intermediate | 892 | 4163 | 4231 | 0.984 |
| Neuro Surgical Intensive Care Unit (Neuro SICU) | 288 | 1054 | 1098 | 0.960 |
| Neuro Stepdown | 219 | 870 | 636 | 1.366 |
| Surgery/Vascular/Intermediate | 27 | 373 | 330 | 1.130 |
| PACU | 16 | 85 | 76 | 1.129 |
