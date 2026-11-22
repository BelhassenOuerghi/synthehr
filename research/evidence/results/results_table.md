# Evidence hardening: 5 seeds, patient-level splits, mean ± 95% CI

Seeds: [42, 43, 44, 45, 46]. Splits are patient-disjoint (the readmission cohort's row-level split leaked 42.7% of test rows). Every figure below replaces a single-seed number from Stages 0 to 2.

## 30-day readmission: XGBoost

| Arm | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|
| TRTR | 0.778 ± 0.007 | 0.379 ± 0.014 | 0.715 ± 0.006 |
| stage0 | 0.690 ± 0.018 | 0.246 ± 0.020 | 0.639 ± 0.053 |
| stage1 | 0.519 ± 0.085 | 0.152 ± 0.035 | 0.544 ± 0.229 |
| dpbaseline | 0.519 ± 0.085 | 0.152 ± 0.035 | 0.544 ± 0.229 |
| fairsample | 0.484 ± 0.039 | 0.137 ± 0.018 | 0.771 ± 0.216 |
| fairtrain | 0.478 ± 0.070 | 0.133 ± 0.028 | 0.726 ± 0.373 |

## Length of stay: RandomForest

| Arm | R² | RMSE (d) | MAE (d) |
|---|---|---|---|
| TRTR | 0.186 ± 0.001 | 3.676 ± 0.003 | 2.358 ± 0.001 |
| stage0 | 0.009 ± 0.083 | 4.052 ± 0.166 | 2.709 ± 0.380 |
| stage1 | -0.671 ± 1.481 | 5.063 ± 2.007 | 3.811 ± 2.657 |
| dpbaseline | -0.671 ± 1.481 | 5.063 ± 2.007 | 3.811 ± 2.657 |
| fairsample | -0.728 ± 1.701 | 5.104 ± 2.240 | 3.889 ± 2.778 |
| fairtrain | -1.416 ± 1.875 | 6.088 ± 2.409 | 5.123 ± 3.249 |

## Bed demand: Poisson-AR

| Arm | MAE | RMSE |
|---|---|---|
| TRTR | 1.475 ± 0.000 | 1.826 ± 0.000 |
| stage0 | 1.475 ± 0.005 | 1.826 ± 0.001 |
| stage1 | 1.460 ± 0.005 | 1.838 ± 0.006 |

## Membership inference (0.5 = no leakage)

| Task | Arm | DCR attack | kNN attack |
|---|---|---|---|
| los | stage0 | 0.498 ± 0.006 | 0.501 ± 0.014 |
| los | stage1 | 0.497 ± 0.006 | 0.502 ± 0.010 |
| los | fairsample | 0.497 ± 0.006 | 0.501 ± 0.006 |
| readmission | stage0 | 0.501 ± 0.006 | 0.505 ± 0.007 |
| readmission | stage1 | 0.503 ± 0.006 | 0.500 ± 0.006 |
| readmission | fairsample | 0.502 ± 0.006 | 0.504 ± 0.006 |

## Run-to-run instability: does DP make the generator unpredictable?

| Task | Metric | sd(stage0, non-private) | sd(stage1, DP) | inflation |
|---|---|---|---|---|
| readmission | roc_auc | 0.014 | 0.068 | **4.8×** |
| los | r2 | 0.066 | 1.193 | **17.9×** |

## Do the claims survive seed noise? (paired t-test across seeds)

| Task | Comparison | Δ | p | verdict |
|---|---|---|---|---|
| readmission (roc_auc) | cost_of_synthesis (TRTR vs stage0) | +0.089 | 0.0002 | **SIGNIFICANT** |
| readmission (roc_auc) | cost_of_privacy (stage0 vs stage1) | +0.171 | 0.0053 | **SIGNIFICANT** |
| readmission (roc_auc) | cost_of_fair_sampling (stage1 vs fairsample) | +0.035 | 0.2683 | within noise |
| readmission (roc_auc) | cost_of_fair_training (stage1 vs fairtrain) | +0.041 | 0.2360 | within noise |
| readmission (roc_auc) | NEGATIVE CONTROL (stage1 vs independent DP replicate) | +0.069 | 0.1658 | within noise |
| los (r2) | cost_of_synthesis (TRTR vs stage0) | +0.176 | 0.0042 | **SIGNIFICANT** |
| los (r2) | cost_of_privacy (stage0 vs stage1) | +0.681 | 0.2789 | within noise |
| los (r2) | cost_of_fair_sampling (stage1 vs fairsample) | +0.056 | 0.5291 | within noise |
| los (r2) | cost_of_fair_training (stage1 vs fairtrain) | +0.745 | 0.5178 | within noise |
| los (r2) | NEGATIVE CONTROL (stage1 vs independent DP replicate) | +0.575 | 0.5259 | within noise |
