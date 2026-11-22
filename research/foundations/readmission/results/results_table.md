# Replication Results vs Upstream Benchmark

## Headline comparison

| Study | Database | Model | ROC-AUC | Notes |
|---|---|---|---|---|
| Blinder (2019), upstream | MIMIC-III | XGBoost, 5-fold CV | **0.82** | Readmission label only; MICU subset; SAPS-II/SOFA features |
| This replication | MIMIC-IV v3.1 | XGBoost, 5-fold CV | **0.7780 ± 0.0044** | Readmission OR 30-day death; all ICU units; no SAPS-II/SOFA |
| Difference (ours − upstream) | | | **-0.0420** | |

## This replication in detail

| Metric | Value |
|---|---|
| Cohort size (ICU stays) | 87,041 |
| Unique patients | 60,537 |
| Positive cases (30-day outcome) | 12,072 |
| Positive rate | 0.1387 |
| &nbsp;&nbsp;of which ICU readmissions | 4,810 |
| &nbsp;&nbsp;of which deaths within 30 d | 7,851 |
| CV ROC-AUC (mean ± std) | 0.7780 ± 0.0044 |
| Out-of-fold ROC-AUC | 0.7780 |
| CV PR-AUC (mean ± std) | 0.3821 ± 0.0099 |
| Out-of-fold PR-AUC | 0.3816 |
| Accuracy @ 0.50 | 0.7142 |
| Precision @ 0.50 | 0.2850 |
| Recall @ 0.50 | 0.7027 |
| F1 @ 0.50 | 0.4055 |

Per-fold ROC-AUC: 0.7751, 0.7766, 0.7723, 0.7846, 0.7815.

Tuned hyperparameters: `{'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 300}` (grid-search best CV ROC-AUC = 0.7772).
