# Stage 1: privacy to utility results (TRTR vs TSTR(ε), real test set)

> Single training run (seed 42) on patient-disjoint splits, kept for provenance. The authoritative numbers are the five-seed means and 95% intervals in [research/evidence](../../evidence): a single training run is not representative of its recipe.

## Length of Stay (regression): RandomForest

| Arm | RMSE (d) | MAE (d) | R² | MIA AUC (DCR / kNN) |
|---|---|---|---|---|
| TRTR (ceiling) | 3.675 | 2.359 | 0.186 | n/a |
| Stage-0 TSTR (ε = ∞) | 3.962 | 2.387 | 0.054 | n/a |
| TSTR (ε = 1.0) | 5.038 | 2.973 | -0.530 | 0.501 / 0.500 |
| TSTR (ε = 5.0) | 4.625 | 3.833 | -0.290 | 0.497 / 0.492 |
| TSTR (ε = 10.0) | 4.228 | 3.108 | -0.078 | 0.497 / 0.497 |

## 30-day Readmission (classification): XGBoost

| Arm | ROC-AUC | PR-AUC | Accuracy | MIA AUC (DCR / kNN) |
|---|---|---|---|---|
| TRTR (ceiling) | 0.770 | 0.361 | 0.718 | n/a |
| Stage-0 TSTR (ε = ∞) | 0.707 | 0.262 | 0.616 | n/a |
| TSTR (ε = 1.0) | 0.497 | 0.134 | 0.655 | 0.493 / 0.505 |
| TSTR (ε = 5.0) | 0.515 | 0.140 | 0.383 | 0.497 / 0.506 |
| TSTR (ε = 10.0) | 0.503 | 0.139 | 0.799 | 0.498 / 0.507 |

## Bed demand (forecast): Poisson-AR

| Arm | MAE | RMSE |
|---|---|---|
| TRTR (ceiling) | 1.475 | 1.826 |
| Stage-0 TSTR (ε = ∞) | 1.472 | 1.827 |
| TSTR (ε = 1.0) | 1.457 | 1.841 |
| TSTR (ε = 5.0) | 1.458 | 1.839 |
| TSTR (ε = 10.0) | 1.458 | 1.839 |
