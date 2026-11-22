"""
train.py, Stacking-ensemble training (scikit-learn) for ICU LOS regression.

Replicates the upstream R methodology (caret + caretEnsemble) in Python:

  Level-0 base learners (each wrapped in its own preprocessing pipeline):
    - LinearRegression                       (upstream: lm)
    - RandomForestRegressor, mtry in {5..10} (upstream: ranger, variance,
      min.node.size = 5). The mtry (max_features) value is chosen by an
      internal grid search, mirroring caret's tuneGrid selection.

  Level-1 meta-learner (trained on out-of-fold base predictions via the
  StackingRegressor's internal 5-fold CV):
    - RandomForestRegressor, max_features = 2 (upstream: ranger, mtry = 2)

The final model is fitted on the train+val partitions (5-fold CV inside the
stack replaces a separate validation set, as upstream did) and held-out test
is untouched here, it is scored by evaluate.py.

Reads : data/derived/{cohort,features}.parquet
Writes: models/stacking_model.joblib, data/derived/splits.parquet
Usage : python src/train.py   (or: make train)
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from data import build_preprocessor, load_config, load_dataset, make_splits, xy


def tune_base_rf(X, y, config):
    """Grid-search the base RF's mtry (max_features) with internal CV.

    Uses a lighter forest (fewer trees) purely to pick max_features, then the
    chosen value is used to build the full-size forest for the stack.
    """
    m = config["model"]
    rf_cfg = m["base_rf"]
    grid = rf_cfg["max_features"]
    print(f"Tuning base RF max_features over {grid} "
          f"({rf_cfg['tune_folds']}-fold CV, {rf_cfg['tune_estimators']} trees)...")

    pipe = Pipeline([
        ("prep", build_preprocessor(scale_numeric=False)),
        ("rf", RandomForestRegressor(
            n_estimators=int(rf_cfg["tune_estimators"]),
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            random_state=int(m["random_state"]),
            n_jobs=-1,
        )),
    ])
    gs = GridSearchCV(
        pipe,
        {"rf__max_features": grid},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=int(rf_cfg["tune_folds"]), shuffle=True,
                 random_state=int(m["random_state"])),
        n_jobs=1,  # inner RF already parallelised across trees
    )
    gs.fit(X, y)
    best = gs.best_params_["rf__max_features"]
    print(f"  best max_features = {best} "
          f"(CV RMSE = {-gs.best_score_:.3f} days)")
    return int(best)


def build_stack(config, base_max_features):
    """Assemble the StackingRegressor (LinearRegression + RF base, RF meta)."""
    m = config["model"]
    seed = int(m["random_state"])
    rf_cfg = m["base_rf"]
    meta_cfg = m["meta_rf"]

    lr_pipe = Pipeline([
        ("prep", build_preprocessor(scale_numeric=True)),
        ("lr", LinearRegression()),
    ])
    rf_pipe = Pipeline([
        ("prep", build_preprocessor(scale_numeric=False)),
        ("rf", RandomForestRegressor(
            n_estimators=int(rf_cfg["n_estimators"]),
            max_features=base_max_features,
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    meta = RandomForestRegressor(
        n_estimators=int(meta_cfg["n_estimators"]),
        max_features=int(meta_cfg["max_features"]),
        min_samples_leaf=int(meta_cfg["min_samples_leaf"]),
        random_state=seed,
        n_jobs=-1,
    )
    return StackingRegressor(
        estimators=[("lr", lr_pipe), ("rf", rf_pipe)],
        final_estimator=meta,
        cv=KFold(n_splits=int(m["cv_folds"]), shuffle=True, random_state=seed),
        n_jobs=1,
        passthrough=False,
    )


def main():
    t0 = time.time()
    config = load_config()
    df = load_dataset(config)
    masks = make_splits(df, config)

    # Persist the split assignment so evaluate.py / visualize.py reuse it exactly.
    split_col = np.where(masks["train"], "train",
                         np.where(masks["val"], "val", "test"))
    os.makedirs(config["derived_dir"], exist_ok=True)
    pd.DataFrame({"stay_id": df["stay_id"].values, "split": split_col}) \
        .to_parquet(f"{config['derived_dir']}/splits.parquet", index=False)

    dev_mask = masks["train"] | masks["val"]
    dev = df[dev_mask]
    n_test = int(masks["test"].sum())
    print(f"Split (patient-level, seed {config['split']['random_state']}): "
          f"train={int(masks['train'].sum()):,} val={int(masks['val'].sum()):,} "
          f"test={n_test:,}")
    print(f"Fitting final model on train+val = {len(dev):,} stays; "
          f"test held out for evaluate.py.")

    X_dev, y_dev = xy(dev)
    best_mf = tune_base_rf(X_dev, y_dev, config)

    print("Fitting stacking ensemble (5-fold internal CV for meta features)...")
    stack = build_stack(config, best_mf)
    stack.fit(X_dev, y_dev)

    os.makedirs(config["models_dir"], exist_ok=True)
    out = f"{config['models_dir']}/stacking_model.joblib"
    joblib.dump({"model": stack, "base_max_features": best_mf}, out)
    print(f"Saved {out}")
    print(f"Training complete in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
