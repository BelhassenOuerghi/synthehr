# Why Python instead of R

The upstream project (`BelhassenOuerghi/Length-of-Stay-Prediction`) is written in R using `caret`, `caretEnsemble`, and `ranger`. This replication uses **Python** instead. This document explains the decision and demonstrates that the scientific methodology is fully preserved.

---

## 1. Motivation

| Concern | R | Python |
|---|---|---|
| ML ecosystem maturity | `caret` (aging, largely superseded) | `scikit-learn` (active, industry standard) |
| Large file handling | Requires loading into memory | DuckDB reads `.csv.gz` natively and lazily |
| Extensibility | Adding deep learning or SHAP is cumbersome | Straightforward (`torch`, `shap`, `mlflow`) |
| Dependency management | `renv`, fragile across OS | `pip` + `requirements.txt`, portable |
| Community & maintenance | Declining in ML research | Dominant in ML and data science |

---

## 2. Component-by-component equivalence

### 2.1 Stacking ensemble

| Upstream R | Python equivalent | Notes |
|---|---|---|
| `caretList(lm, ranger)` | `StackingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor())])` | Same two-learner stacking architecture |
| `caretStack(ranger)` | `final_estimator=RandomForestRegressor()` | Same RF meta-learner |
| `trainControl(method="cv", number=5, savePredictions="final")` | `KFold(n_splits=5, shuffle=True)` with `cross_val_predict` | 5-fold CV, out-of-fold predictions for stacking |
| `metric="RMSE"` | `scoring="neg_root_mean_squared_error"` | Same optimisation target |

### 2.2 Base learner: Linear Regression

| Upstream R | Python |
|---|---|
| `caretModelSpec(method="lm")` | `sklearn.linear_model.LinearRegression()` |

No hyperparameter tuning in either implementation.

### 2.3 Base learner: Random Forest

| Upstream R | Python |
|---|---|
| `ranger`, `.mtry = c(5:10)` | `RandomForestRegressor`, `max_features` ∈ {5..10} |
| `.splitrule = "variance"` | Default RF variance splitting |
| `.min.node.size = 5` | `min_samples_leaf=5` |

### 2.4 Meta-learner: Random Forest

| Upstream R | Python |
|---|---|
| `ranger`, `mtry = 2` | `RandomForestRegressor`, `max_features=2` |
| `min.node.size ∈ {5,10,15,20}` | `min_samples_leaf` ∈ {5,10,15,20} |
| `splitrule ∈ {variance, extratrees, maxstat}` | `criterion ∈ {"squared_error", "absolute_error"}` (closest available) |

> **Note on `splitrule`:** scikit-learn's RF does not expose all three `ranger` split rules. `squared_error` maps to variance; `absolute_error` is the closest proxy to `maxstat`. This is a minor implementation difference and does not affect the core methodology.

### 2.5 Metrics

| Upstream R (`MLmetrics`) | Python (`sklearn.metrics`) |
|---|---|
| `RMSE(y_pred, y_true)` | `mean_squared_error(y_true, y_pred, squared=False)` |
| `MAE(y_pred, y_true)` | `mean_absolute_error(y_true, y_pred)` |
| `R2_Score(y_pred, y_true)` | `r2_score(y_true, y_pred)` |

### 2.6 Prediction clipping

Both implementations clip predictions to [0, 21] days before SLOS computation:

```r
# R (upstream)
mutate(predicted = case_when(predicted < 0 ~ 0, predicted > 21 ~ 21, TRUE ~ predicted))
```

```python
# Python (this repo)
predicted_clipped = predicted.clip(0, 21)
```

### 2.7 Visualisations

| Upstream R | Python |
|---|---|
| `ggplot2` predicted vs observed | `matplotlib` scatter + LOESS-style smoothing |
| `ems::funnel()` | `matplotlib` funnel plot with control limits |

---

## 3. What is NOT identical (and why it does not matter)

1. **Random number generator:** R and Python use different RNG algorithms. With the same seed, split assignments and tree structures will differ, but the methodology is the same.
2. **Ranger vs sklearn RF:** `ranger` is highly optimised for speed; `RandomForestRegressor` is the standard Python equivalent. Both implement CART regression trees with bootstrap sampling.
3. **`splitrule=extratrees`:** not available in scikit-learn; excluded from the meta-learner grid search.

These differences affect reproducibility at the implementation level but **not at the methodological level**. The published results are comparable, any performance difference reflects dataset (MIMIC-III vs upstream proprietary data) not the language switch.

---

## 4. Conclusion

The Python implementation is a **faithful methodological replication** of the upstream R workflow. The switch introduces no scientific loss and yields a more maintainable, extensible, and portable codebase.
