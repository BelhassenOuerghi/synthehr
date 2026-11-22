"""
extract.py, Cohort and feature extraction from MIMIC-IV v3.1 using DuckDB.

This module was originally written against MIMIC-III (UPPERCASE schema,
DOB-derived age, CareVue+MetaVision itemids). It has been ADAPTED to
MIMIC-IV v3.1, which differs in several material ways:

  * Lowercase table and column names.
  * ICU LOS is precomputed in `icu/icustays.los` (fractional days), no
    need to recompute it from outtime - intime.
  * The ICU-stay key is `stay_id` (was `icustay_id`); age is `anchor_age`
    on `hosp/patients` (no shifted DOB arithmetic).
  * `chartevents` is MetaVision-only, so vital-sign itemids live in the
    2200xx / 2237xx ranges (no CareVue 2xx/6xx ids).
  * `labevents` carries `hadm_id`, so labs join to the cohort by hadm_id.

The big event tables (chartevents 3.5 GB, labevents 2.6 GB gzipped) are
read directly from `.csv.gz` by DuckDB with the itemid filter and the
first-24h time window pushed down into SQL. They are NEVER loaded whole
into pandas. Aggregation happens in DuckDB with a bounded memory limit.

Outputs (both gitignored, derived per-patient data must not be committed):

  data/derived/cohort.parquet
    One row per ICU stay: subject_id, hadm_id, stay_id, intime, outtime,
    UnitCode, UnitLengthStay, UnitLengthStay_trunc

  data/derived/features.parquet
    stay_id + the model predictors (demographics, admission context,
    first-24h vital-sign aggregates, first-24h laboratory aggregates).

Usage:
    /path/to/venv/bin/python src/extract.py
    # or:  make extract
"""

import argparse
import os

import duckdb
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Feature dictionary: MIMIC-IV v3.1 itemids
# ---------------------------------------------------------------------------
# Vital signs live in icu/chartevents (verified against icu/d_items). Because
# MIMIC-IV is MetaVision-only, each concept maps to a small set of MetaVision
# itemids. For each vital we aggregate mean / min / max over the first 24 h.
#
#   itemid  label (d_items.label)                 concept
#   220045  Heart Rate                            heart_rate      (bpm)
#   220210  Respiratory Rate                      resp_rate       (insp/min)
#   220277  O2 saturation pulseoxymetry           spo2            (%)
#   223762  Temperature Celsius                   temp_c          (degC, direct)
#   223761  Temperature Fahrenheit                temp_c          (degC, F->C)
#   220052  Arterial Blood Pressure mean          map             (mmHg)
#   220181  Non Invasive Blood Pressure mean      map             (mmHg)
#   225312  ART BP Mean                           map             (mmHg)
#
# Physiologic sanity bounds drop obviously erroneous charted values (e.g. a
# heart rate of 10109 or 0) before aggregation, so a single bad reading cannot
# dominate a MAX/MIN.
VITAL_BOUNDS = {
    "heart_rate": (10, 300),
    "resp_rate": (1, 80),
    "spo2": (10, 100),
    "temp_c": (25, 45),
    "map": (10, 250),
}

# Laboratory analytes live in hosp/labevents (verified against hosp/d_labitems).
# For each we take the mean value over the first 24 h.
#
#   itemid  label (d_labitems.label)   concept       bounds
#   50912   Creatinine                 creatinine    (0, 50)   mg/dL
#   51301   White Blood Cells          wbc           (0, 200)  K/uL
#   51300   WBC Count                  wbc           (0, 200)  K/uL
#   50931   Glucose                    glucose       (0, 2000) mg/dL
#   50983   Sodium                     sodium        (80, 200) mEq/L
#   50971   Potassium                  potassium     (0, 15)   mEq/L
#   50882   Bicarbonate                bicarbonate   (0, 60)   mEq/L
#   51006   Urea Nitrogen              bun           (0, 300)  mg/dL
#   51221   Hematocrit                 hematocrit    (0, 100)  %
#   51265   Platelet Count             platelets     (0, 2000) K/uL
LAB_ITEMS = {
    "creatinine": ([50912], (0, 50)),
    "wbc": ([51301, 51300], (0, 200)),
    "glucose": ([50931], (0, 2000)),
    "sodium": ([50983], (80, 200)),
    "potassium": ([50971], (0, 15)),
    "bicarbonate": ([50882], (0, 60)),
    "bun": ([51006], (0, 300)),
    "hematocrit": ([51221], (0, 100)),
    "platelets": ([51265], (0, 2000)),
}

# Explicit schemas keep DuckDB from scanning the whole 3.5 GB chartevents file
# to infer types, and pin `value` as VARCHAR (it contains free text with stray
# quotes, e.g. '1" Packing', which requires strict_mode=false).
CHARTEVENTS_COLS = {
    "subject_id": "BIGINT", "hadm_id": "BIGINT", "stay_id": "BIGINT",
    "caregiver_id": "BIGINT", "charttime": "TIMESTAMP", "storetime": "TIMESTAMP",
    "itemid": "BIGINT", "value": "VARCHAR", "valuenum": "DOUBLE",
    "valueuom": "VARCHAR", "warning": "BIGINT",
}
LABEVENTS_COLS = {
    "labevent_id": "BIGINT", "subject_id": "BIGINT", "hadm_id": "BIGINT",
    "specimen_id": "BIGINT", "itemid": "BIGINT", "order_provider_id": "VARCHAR",
    "charttime": "TIMESTAMP", "storetime": "TIMESTAMP", "value": "VARCHAR",
    "valuenum": "DOUBLE", "valueuom": "VARCHAR", "ref_range_lower": "DOUBLE",
    "ref_range_upper": "DOUBLE", "flag": "VARCHAR", "priority": "VARCHAR",
    "comments": "VARCHAR",
}


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _cols_literal(schema):
    return "{" + ", ".join(f"'{c}': '{t}'" for c, t in schema.items()) + "}"


def connect(config):
    """Open an in-memory DuckDB connection with bounded resources."""
    mimic_dir = os.path.expanduser(config["mimic_dir"])
    if not os.path.isdir(mimic_dir):
        raise FileNotFoundError(f"MIMIC-IV directory not found: {mimic_dir}")
    duck = config.get("duckdb", {})
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{duck.get('memory_limit', '6GB')}';")
    con.execute(f"SET threads={int(duck.get('threads', 4))};")
    return con, mimic_dir


def _path(mimic_dir, rel):
    p = os.path.join(mimic_dir, rel)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing required MIMIC-IV file: {p}")
    return p.replace("'", "''")


def extract_cohort(con, mimic_dir, config):
    """Build the ICU-stay cohort (small, fast) from icustays + patients + admissions.

    Inclusion: adults (anchor_age >= min_age), ICU stay >= min_icu_hours,
    non-null los. UnitLengthStay is icustays.los directly; UnitLengthStay_trunc
    is los clipped to [0, los_clip_days]. UnitCode is first_careunit.
    """
    print("Extracting cohort (icustays + patients + admissions)...")
    coh = config["cohort"]
    min_age = int(coh["min_age"])
    min_los_days = float(coh["min_icu_hours"]) / 24.0
    clip = float(coh["los_clip_days"])

    icu = _path(mimic_dir, "icu/icustays.csv.gz")
    pat = _path(mimic_dir, "hosp/patients.csv.gz")
    adm = _path(mimic_dir, "hosp/admissions.csv.gz")

    q = f"""
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        i.first_careunit          AS UnitCode,
        i.los                     AS UnitLengthStay,
        LEAST(GREATEST(i.los, 0), {clip}) AS UnitLengthStay_trunc,
        p.anchor_age              AS age,
        p.gender                  AS gender,
        a.admission_type          AS admission_type,
        -- hospital days elapsed before ICU admission (known at ICU intime)
        GREATEST(EXTRACT(EPOCH FROM (i.intime - a.admittime)) / 86400.0, 0) AS pre_icu_los_days
    FROM read_csv_auto('{icu}') i
    JOIN read_csv_auto('{pat}') p ON i.subject_id = p.subject_id
    LEFT JOIN read_csv_auto('{adm}') a ON i.hadm_id = a.hadm_id
    WHERE p.anchor_age >= {min_age}
      AND i.los IS NOT NULL
      AND i.los >= {min_los_days}
    ORDER BY i.stay_id
    """
    df = con.execute(q).fetchdf()
    df.columns = [c.lower() for c in df.columns]

    sample_n = config["cohort"].get("sample_n")
    if sample_n:
        n = min(int(sample_n), len(df))
        df = df.sample(n=n, random_state=int(config["split"]["random_state"])).reset_index(drop=True)
        print(f"  Sampled {n:,} of the eligible stays (seed "
              f"{config['split']['random_state']}).")
    print(f"  Cohort: {len(df):,} ICU stays, {df['subject_id'].nunique():,} subjects.")
    return df


def extract_vitals(con, mimic_dir, cohort_df, config):
    """First-24h vital-sign aggregates from icu/chartevents (pushed-down in SQL)."""
    print("Aggregating first-24h vitals from chartevents (this takes a few minutes)...")
    win = int(config["cohort"].get("obs_window_hours", 24))
    ce = _path(mimic_dir, "icu/chartevents.csv.gz")
    con.register("_cohort", cohort_df[["stay_id", "intime"]])

    def bounded(itemids, lo, hi, expr="ce.valuenum"):
        ids = ",".join(str(i) for i in itemids)
        return (f"CASE WHEN ce.itemid IN ({ids}) AND ({expr}) BETWEEN {lo} AND {hi} "
                f"THEN ({expr}) END")

    hr = bounded([220045], *VITAL_BOUNDS["heart_rate"])
    rr = bounded([220210], *VITAL_BOUNDS["resp_rate"])
    spo2 = bounded([220277], *VITAL_BOUNDS["spo2"])
    # Temperature: 223762 is already Celsius; 223761 is Fahrenheit -> convert.
    temp = (
        "CASE "
        "WHEN ce.itemid = 223762 AND ce.valuenum BETWEEN 25 AND 45 THEN ce.valuenum "
        "WHEN ce.itemid = 223761 AND ce.valuenum BETWEEN 77 AND 113 "
        "THEN (ce.valuenum - 32) * 5.0/9.0 END"
    )
    mp = bounded([220052, 220181, 225312], *VITAL_BOUNDS["map"])

    q = f"""
    SELECT
        c.stay_id,
        AVG({hr})   AS hr_mean,  MIN({hr})   AS hr_min,  MAX({hr})   AS hr_max,
        AVG({rr})   AS rr_mean,  MIN({rr})   AS rr_min,  MAX({rr})   AS rr_max,
        AVG({spo2}) AS spo2_mean,MIN({spo2}) AS spo2_min,MAX({spo2}) AS spo2_max,
        AVG({temp}) AS temp_mean,MIN({temp}) AS temp_min,MAX({temp}) AS temp_max,
        AVG({mp})   AS map_mean, MIN({mp})   AS map_min, MAX({mp})   AS map_max
    FROM read_csv_auto('{ce}', columns={_cols_literal(CHARTEVENTS_COLS)},
                       strict_mode=false, ignore_errors=true) ce
    JOIN _cohort c ON ce.stay_id = c.stay_id
    WHERE ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '{win} hours'
      AND (ce.warning IS NULL OR ce.warning != 1)
      AND ce.valuenum IS NOT NULL
      AND ce.itemid IN (220045, 220210, 220277, 223762, 223761, 220052, 220181, 225312)
    GROUP BY c.stay_id
    """
    df = con.execute(q).fetchdf()
    df.columns = [c.lower() for c in df.columns]
    con.unregister("_cohort")
    print(f"  Vitals aggregated for {len(df):,} stays.")
    return df


def extract_labs(con, mimic_dir, cohort_df, config):
    """First-24h laboratory aggregates from hosp/labevents (pushed-down in SQL).

    labevents has no stay_id, so we join on hadm_id and attribute a lab draw to
    a stay only when its charttime falls inside that stay's first-24h window.
    """
    print("Aggregating first-24h labs from labevents (this takes a few minutes)...")
    win = int(config["cohort"].get("obs_window_hours", 24))
    le = _path(mimic_dir, "hosp/labevents.csv.gz")
    con.register("_cohort", cohort_df[["stay_id", "hadm_id", "intime"]])

    all_ids = sorted({i for ids, _ in LAB_ITEMS.values() for i in ids})
    id_filter = ",".join(str(i) for i in all_ids)
    aggs = []
    for name, (ids, (lo, hi)) in LAB_ITEMS.items():
        ids_sql = ",".join(str(i) for i in ids)
        aggs.append(
            f"AVG(CASE WHEN le.itemid IN ({ids_sql}) AND le.valuenum BETWEEN {lo} AND {hi} "
            f"THEN le.valuenum END) AS {name}"
        )
    agg_sql = ",\n        ".join(aggs)

    q = f"""
    SELECT
        c.stay_id,
        {agg_sql}
    FROM read_csv_auto('{le}', columns={_cols_literal(LABEVENTS_COLS)},
                       strict_mode=false, ignore_errors=true) le
    JOIN _cohort c ON le.hadm_id = c.hadm_id
        AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '{win} hours'
    WHERE le.valuenum IS NOT NULL
      AND le.itemid IN ({id_filter})
    GROUP BY c.stay_id
    """
    df = con.execute(q).fetchdf()
    df.columns = [c.lower() for c in df.columns]
    con.unregister("_cohort")
    print(f"  Labs aggregated for {len(df):,} stays.")
    return df


# Feature column groups (used here and by src/data.py for the model).
NUMERIC_FEATURES = (
    ["age", "pre_icu_los_days"]
    + ["hr_mean", "hr_min", "hr_max", "rr_mean", "rr_min", "rr_max",
       "spo2_mean", "spo2_min", "spo2_max", "temp_mean", "temp_min", "temp_max",
       "map_mean", "map_min", "map_max"]
    + list(LAB_ITEMS.keys())
)
CATEGORICAL_FEATURES = ["gender", "admission_type", "unitcode"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(cohort_df, vitals_df, labs_df):
    """Assemble the model feature table keyed by stay_id (left joins on cohort)."""
    base = cohort_df[["stay_id", "age", "pre_icu_los_days", "gender",
                      "admission_type", "unitcode"]].copy()
    feats = base.merge(vitals_df, on="stay_id", how="left")
    feats = feats.merge(labs_df, on="stay_id", how="left")
    # Order: stay_id + declared feature columns.
    ordered = ["stay_id"] + [c for c in ALL_FEATURES if c in feats.columns]
    return feats[ordered]


def print_inventory(cohort_df, features_df):
    print("\nExtraction summary")
    print("=" * 72)
    print(f"cohort rows   : {len(cohort_df):,}")
    print(f"feature rows  : {len(features_df):,}")
    print(f"feature cols  : {len(features_df.columns) - 1} predictors")
    los = cohort_df["unitlengthstay_trunc"]
    print(f"\nLOS_trunc (days): mean={los.mean():.2f} median={los.median():.2f} "
          f"std={los.std():.2f} min={los.min():.2f} max={los.max():.2f}")
    print("\nPer-feature null rate (missing physiology is expected & imputed at train):")
    print("-" * 72)
    for col in features_df.columns:
        if col == "stay_id":
            continue
        nr = features_df[col].isna().mean() * 100
        print(f"  {col:<20} {nr:6.2f}% null")


def save(cohort_df, features_df, config):
    out = config["derived_dir"]
    os.makedirs(out, exist_ok=True)
    cohort_df.to_parquet(f"{out}/cohort.parquet", index=False)
    features_df.to_parquet(f"{out}/features.parquet", index=False)
    print(f"\nSaved {out}/cohort.parquet and {out}/features.parquet")


def main():
    print("MIMIC-IV v3.1, cohort & feature extraction")
    config = load_config()
    print(f"MIMIC dir: {config['mimic_dir']}")
    con, mimic_dir = connect(config)

    cohort_df = extract_cohort(con, mimic_dir, config)
    vitals_df = extract_vitals(con, mimic_dir, cohort_df, config)
    labs_df = extract_labs(con, mimic_dir, cohort_df, config)
    features_df = build_features(cohort_df, vitals_df, labs_df)

    save(cohort_df, features_df, config)
    print_inventory(cohort_df, features_df)
    print("\nExtraction complete.")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Extract MIMIC-IV LOS cohort/features").parse_args()
    main()
