# Stage 2: fairness to utility results (real test set)

> Single training run (seed 42) on patient-disjoint splits, kept for provenance. The authoritative numbers are the five-seed means and 95% intervals in [research/evidence](../../evidence): a single training run is not representative of its recipe.

## 30-day readmission: XGBoost (sensitive: gender, insurance)

| Arm | ROC-AUC | PR-AUC | AUC gap (gen / ins) | DP diff (gen / ins) | EO diff (gen / ins) |
|---|---|---|---|---|---|
| TRTR | 0.770 | 0.361 | 0.010 / 0.056 | 0.028 / 0.314 | 0.027 / 0.284 |
| stage0 | 0.690 | 0.234 | 0.019 / 0.024 | 0.058 / 0.333 | 0.072 / 0.422 |
| stage1 | 0.503 | 0.139 | 0.012 / 0.096 | 0.005 / 0.137 | 0.024 / 0.139 |
| dpbaseline | 0.401 | 0.105 | 0.008 / 0.074 | 0.006 / 0.299 | 0.016 / 0.293 |
| fairsample | 0.451 | 0.119 | 0.000 / 0.114 | 0.051 / 0.088 | 0.053 / 0.091 |
| fairtrain | 0.451 | 0.117 | 0.030 / 0.093 | 0.040 / 0.174 | 0.075 / 0.247 |

## Length of stay: RandomForest (sensitive: gender)

| Arm | RMSE (d) | R² | RMSE gap | bias gap |
|---|---|---|---|---|
| TRTR | 3.675 | 0.186 | 0.172 | 0.188 |
| stage0 | 3.962 | 0.054 | 0.206 | 0.128 |
| stage1 | 4.228 | -0.078 | 0.159 | 0.338 |
| dpbaseline | 4.423 | -0.180 | 0.419 | 0.621 |
| fairsample | 4.447 | -0.192 | 0.416 | 0.372 |
| fairtrain | 8.852 | -3.724 | 1.295 | 1.633 |

## Bed demand: Poisson-AR (DP pass-through)

| Arm | MAE | RMSE |
|---|---|---|
| TRTR | 1.475 | 1.826 |
| TSTR (stage 2) | 1.458 | 1.839 |

## Membership inference: canonical arm (`fairsample`)

| Task | DCR AUC | kNN AUC | near-dup@0.1 |
|---|---|---|---|
| los | 0.497 | 0.503 | 0.000 |
| readmission | 0.499 | 0.501 | 0.000 |
