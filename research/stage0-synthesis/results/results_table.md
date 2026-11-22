# Utility results: TRTR vs TSTR (real test set)

> Single training run (seed 42) on patient-disjoint splits, kept for provenance. The authoritative numbers are the five-seed means and 95% intervals in [research/evidence](../../evidence): a single training run is not representative of its recipe.

## Length of Stay (regression): RandomForest

| Arm | RMSE (d) | MAE (d) | R² |
|---|---|---|---|
| TRTR (train real) | 3.675 | 2.359 | 0.186 |
| TSTR (train synthetic) | 3.962 | 2.387 | 0.054 |
| Δ R² (TSTR − TRTR) | | | -0.132 |

## 30-day Readmission (classification): XGBoost

| Arm | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|
| TRTR (train real) | 0.770 | 0.361 | 0.718 |
| TSTR (train synthetic) | 0.690 | 0.234 | 0.711 |
| Δ ROC-AUC (TSTR − TRTR) | -0.080 | | |

## Bed demand (forecast): Poisson-AR

| Arm | MAE | RMSE |
|---|---|---|
| TRTR (train real) | 1.475 | 1.826 |
| TSTR (train synthetic) | 1.472 | 1.827 |
| Δ MAE (TSTR − TRTR) | -0.003 | |
