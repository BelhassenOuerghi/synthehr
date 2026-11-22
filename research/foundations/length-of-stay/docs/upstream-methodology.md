# Upstream Methodology (Length-of-Stay-Prediction): Detailed Technical Notes

This document reverse-engineers and explains the methodology implemented in the upstream repository:

- Upstream (R): https://github.com/igor-peres/ICU-Length-of-Stay-Prediction
- Method reference: Peres et al. (2022), *Data-driven methodology to predict the ICU length of stay* (Anaesthesia Critical Care & Pain Medicine).

It is written to help us **replicate the experiment faithfully** and then **port it to MIMIC-IV v3.1** with minimal ambiguity. (Note: the pipeline was originally scaffolded for MIMIC-III and has since been adapted to MIMIC-IV, see the main README.)

> Scope note
> - The upstream README indicates the “second version” of the model is in `Stacking_NumericLOS_V2.0/`.
> - The analysis below focuses primarily on that folder, plus the unit-level SLOS computation implemented in the testing script.

---

## 1. What is being predicted?

### 1.1 Prediction task

The upstream project implements a **supervised regression** task:

- **Goal:** predict ICU **Length of Stay (LOS)** in days.
- **Type:** *numeric* LOS prediction (continuous regression).

### 1.2 Label / target variable

Across scripts, the target column is:

- `UnitLengthStay_trunc`

This strongly suggests:

- LOS is measured at an ICU “unit stay” level (an ICU admission/episode).
- LOS has been **truncated** (or capped) before modeling.

#### Truncation and clipping behavior

In `Stacking_NumericLOS_V2.0/Testing.R`, after generating predictions, the code applies:

```r
mutate(predicted = case_when(
  predicted < 0 ~ 0,
  predicted > 21 ~ 21,
  TRUE ~ predicted
))
```

So, regardless of whether the label is explicitly truncated, **predictions are clipped to [0, 21]** days for the downstream SLOS computation.

**Implication for replication:**

- We should clarify whether **labels** are also capped at 21, or only **predictions**.
- For faithful parity, at minimum we must clip predictions to the same range before computing SLOS.

---

## 2. What are the inputs (features)?

### 2.1 Predictor set selection

In `Stacking_NumericLOS_V2.0/Training.R`, the feature names are loaded from a CSV:

```r
predictors = read.csv('predictors.csv')
predictors = predictors[,2]

data = data %>%
  select(predictors, UnitLengthStay_trunc)
```

So the model is trained on a **pre-selected list of predictors** maintained outside the code.

**Important:**

- The feature list is treated as authoritative.
- The training script assumes the dataset contains all required columns.

### 2.2 Data dictionary

The repo includes `DataDictionary.pdf`, which appears to document the schema for the stacking model features.

**Replication expectation:**

- Our MIMIC-III feature engineering should create columns that correspond (as closely as possible) to the upstream predictors.
- If some upstream predictors are not available in MIMIC-III, we must:
  - document missingness,
  - implement proxies, or
  - run ablations.

---

## 3. Modeling approach: stacked ensemble in caret

Upstream uses the `caret` and `caretEnsemble` ecosystem.

### 3.1 Packages

Both training and testing scripts load:

- `caret`
- `tidyverse`
- `caretEnsemble`
- `MLmetrics`

Testing additionally uses:

- `ems` (for funnel plots)

### 3.2 Cross-validation protocol

Training uses 5-fold cross-validation:

```r
fitControl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  trim = TRUE,
  savePredictions = "final"
)
```

Key details:

- **5-fold CV**: splits training data into 5 folds.
- `savePredictions = "final"`: stores the out-of-fold predictions for each resample (required for stacking).
- `returnData = FALSE`: avoids storing raw training data inside the model objects.

### 3.3 Base learners (level-0 models)

The upstream stacking model trains a small set of base models:

```r
model_list_complete <- caretList(
  x = training[,-ncol(training)],
  y = training$UnitLengthStay_trunc,
  trControl = fitControl,
  metric = "RMSE",
  tuneList = list(
    lm = caretModelSpec(method = "lm"),
    rf = caretModelSpec(
      method = "ranger",
      tuneGrid = data.frame(
        .mtry = c(5:10),
        .splitrule = "variance",
        .min.node.size = 5
      )
    )
  )
)
```

#### Base learner 1: Linear regression (lm)

- `method = "lm"`
- No explicit hyperparameter tuning.

#### Base learner 2: Random forest (ranger)

- `method = "ranger"`
- Tuning grid:
  - `.mtry` in {5,6,7,8,9,10}
  - `.splitrule = "variance"`
  - `.min.node.size = 5`

**Interpretation:**

- This is a classic regression random forest configuration.
- Tuning is fairly lightweight; only `mtry` varies.

### 3.4 Meta-learner (level-1 model)

The stacked model is trained using `caretStack` with `ranger` as the meta-model:

```r
rfGrid = expand.grid(
  mtry = 2,
  min.node.size = c(5,10,15,20),
  splitrule = c("variance","extratrees","maxstat")
)

SLOS_model <- caretStack(
  model_list_complete,
  trControl = fitControl,
  metric = "RMSE",
  method = "ranger",
  tuneGrid = rfGrid
)
```

Key details:

- Meta-model is also `ranger`.
- Meta tuning grid:
  - `mtry` fixed at 2
  - `min.node.size` ∈ {5, 10, 15, 20}
  - `splitrule` ∈ {variance, extratrees, maxstat}

**Interpretation:**

- With two base learners (lm + rf), the stacked feature space is small.
- Fixing `mtry = 2` means the meta-model can consider both base predictions at each split.

### 3.5 Saved artifacts

Training script saves:

- `SLOS_model_list.RData`, the base learners
- `SLOS_model.RData`, final stacked model

These are intended to be loaded by the testing script.

---

## 4. Testing & performance evaluation

### 4.1 Test dataset

Upstream testing reads:

```r
testing = read.csv("Synthetic_TestingData.csv")
```

So the repository ships with **synthetic test data** matching the expected schema.

### 4.2 Model loading and prediction

```r
load("SLOS_model.RData")
model = SLOS_model
Predicted = data.frame(Predicted = predict(model, newdata = testing))
Observed = data.frame(Observed = testing$UnitLengthStay_trunc)
```

### 4.3 Metrics

Upstream computes:

- RMSE
- MAE
- R²

using `MLmetrics`:

```r
Erro = RMSE(y_pred = Predicted$Predicted, y_true = Observed$Observed)
MAE  = MAE(y_pred = Predicted$Predicted, y_true = Observed$Observed)
R2   = R2_Score(y_pred = Predicted$Predicted, y_true = Observed$Observed)
```

### 4.4 Calibration-style plot

They generate a predicted-vs-observed scatter plot with smoothing and a 45° reference line.

**Replication note:**

- The plot is not a formal calibration method, but it is useful for visual diagnostics.

---

## 5. Unit-level SLOS analysis (operational / benchmarking)

A distinctive part of upstream is the SLOS computation.

### 5.1 Row-level prediction table

Upstream constructs a dataframe with observed and predicted LOS:

```r
df_model_pred = testing %>%
  select(observed = UnitLengthStay_trunc) %>%
  mutate(predicted = predict(model, newdata = testing)) %>%
  mutate(predicted = case_when(
    predicted < 0 ~ 0,
    predicted > 21 ~ 21,
    TRUE ~ predicted
  ))
```

### 5.2 Aggregation by unit

They bind unit identifiers and aggregate:

```r
df_unit_slos = df_model_pred %>%
  bind_cols(testing %>% select(UnitCode)) %>%
  group_by(UnitCode) %>%
  summarise(
    admissoes = n(),
    soma_los_obs = sum(observed),
    soma_los_esp = sum(predicted)
  ) %>%
  ungroup() %>%
  mutate(SLOS = soma_los_obs / soma_los_esp) %>%
  na.omit()
```

Definitions:

- `admissoes`: number of ICU admissions in that unit
- `soma_los_obs`: total observed LOS days in that unit
- `soma_los_esp`: total predicted LOS days in that unit
- `SLOS`: standardized ratio = observed / expected

### 5.3 Overall SLOS

```r
df_model_pred %>% summarise(SLOS_geral = sum(observed) / sum(predicted))
```

### 5.4 Aggregate goodness-of-fit

They compute R² on aggregated totals by unit:

```r
R2 = R2_Score(df_unit_slos$soma_los_esp, df_unit_slos$soma_los_obs)
```

This is a different question than individual-level prediction R²:

- Individual-level R²: how well we predict LOS per patient
- Unit-level R²: how well aggregated expected LOS tracks observed LOS per unit

### 5.5 Funnel plot

They use the `ems` package to produce a funnel plot with:

- `y = SLOS`
- expected and observed totals
- `theta = overall observed / overall expected`

This is a classic quality-monitoring visualization.

---

## 6. Practical replication checklist

To replicate upstream results end-to-end, ensure:

1. **Schema parity**
   - All predictors listed in `predictors.csv` exist in your training/testing tables.
2. **Consistent target**
   - Use the same definition of `UnitLengthStay_trunc`.
3. **Cross-validation**
   - 5-fold CV, `savePredictions = "final"`.
4. **Base models**
   - `lm`
   - `ranger` with `.mtry = 5:10`, splitrule variance, min.node.size 5
5. **Meta model**
   - `ranger` with mtry=2, min.node.size ∈ {5,10,15,20}, splitrule ∈ {variance, extratrees, maxstat}
6. **Metrics**
   - RMSE, MAE, R² using the same functions
7. **SLOS computation**
   - Clip predictions to [0, 21] days before aggregating

---

## 7. Mapping upstream methodology to MIMIC-III

### 7.1 Cohort definition (to be finalized)

For MIMIC-III, the natural unit of analysis is typically the ICU stay (`ICUSTAYS`).

We must decide:

- whether to model **first ICU stay per hospital admission** or all ICU stays,
- inclusion/exclusion criteria (age, care units, length bounds, etc.),
- handling of transfers between units.

### 7.2 Label definition in MIMIC-III

Candidate:

- ICU LOS days = (`OUTTIME` - `INTIME`) in days

Then define:

- `UnitLengthStay_trunc` = min(max(LOS_days, 0), 21) or similar (depending on upstream definition)

We will report both:

- unclipped regression performance
- clipped performance (for SLOS comparability)

### 7.3 Feature engineering

We will implement a **feature spec** that maps each upstream predictor to:

- MIMIC-III source table(s)
- SQL expression / extraction logic
- unit conversions
- missingness handling

This mapping will live in `docs/feature-spec.md`.

---

## 8. Known gaps / ambiguities to resolve

1. **Where is `training` defined in `Training.R`?**
   - The script loads `data.RData` and constructs `data`, but then calls `caretList(x=training[...])`.
   - This suggests `data.RData` may define an object named `training` already, or the script has a minor naming inconsistency.
   - For replication, we should standardize to one object name and document it.

2. **Exact definition of `UnitLengthStay_trunc`**
   - We see clipping in predictions, but we need the upstream rule for the label.

3. **Predictor semantics**
   - We must consult `DataDictionary.pdf` and the shipped synthetic datasets to reproduce feature definitions.

---

## 9. References (upstream code pointers)

- Training: `Stacking_NumericLOS_V2.0/Training.R`
- Testing + SLOS: `Stacking_NumericLOS_V2.0/Testing.R`
- Data dictionary: `DataDictionary.pdf`
- Predictor list: `Stacking_NumericLOS_V2.0/predictors.csv` (in upstream)

