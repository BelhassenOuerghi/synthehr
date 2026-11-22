"""Feature engineering with DuckDB.

Vitals are read from icu/chartevents.csv.gz (3.5 GB) and labs from
hosp/labevents.csv.gz (2.6 GB). Both are scanned DIRECTLY from gzip by DuckDB,
filtered to the cohort's stay_ids / hadm_ids and to the configured itemids, and
aggregated (min/max/mean) inside SQL over the first `feature_window_hours` of
each ICU stay. The large tables are NEVER loaded whole into pandas.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from .config import load_config, mimic_path
from .build_cohort import build_cohort, cohort_summary

# Explicit column typing keeps DuckDB from mis-sniffing, and the tolerant flags
# (strict_mode=false, ignore_errors=true) skip the handful of rows in
# chartevents that carry unescaped double-quotes inside free-text `value`
# fields. Those rows are non-numeric annotations irrelevant to our itemids.
_CHART_COLS = ("{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT',"
               "'caregiver_id':'BIGINT','charttime':'TIMESTAMP','storetime':'TIMESTAMP',"
               "'itemid':'BIGINT','value':'VARCHAR','valuenum':'DOUBLE',"
               "'valueuom':'VARCHAR','warning':'BIGINT'}")
_LAB_COLS = ("{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',"
             "'specimen_id':'BIGINT','itemid':'BIGINT','order_provider_id':'VARCHAR',"
             "'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',"
             "'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',"
             "'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',"
             "'comments':'VARCHAR'}")


def _read_chart(path: str) -> str:
    return (f"read_csv('{path}', columns={_CHART_COLS}, "
            f"strict_mode=false, ignore_errors=true, quote='\"', escape='\"')")


def _read_lab(path: str) -> str:
    return (f"read_csv('{path}', columns={_LAB_COLS}, "
            f"strict_mode=false, ignore_errors=true, quote='\"', escape='\"')")


def _flat(ids_map: dict) -> list[int]:
    out: list[int] = []
    for v in ids_map.values():
        out.extend(v)
    return sorted(set(out))


def build_vitals(cfg: dict, con: duckdb.DuckDBPyConnection) -> None:
    """Aggregate chart-event vitals per ICU stay into temp table `vitals`."""
    v = cfg["vitals"]
    b = cfg["vital_bounds"]
    win = int(cfg["cohort"]["feature_window_hours"])
    chart = mimic_path(cfg, "icu", "chartevents")

    def ids(name: str) -> str:
        return ",".join(map(str, v[name]))

    def agg(name: str, expr: str, stat: str) -> str:
        return f"{stat}({expr}) AS {name}_{stat.split('(')[0].lower()}"

    # Bounded numeric expression per vital.
    def bounded(name: str) -> str:
        lo, hi = b[name]
        return (f"CASE WHEN itemid IN ({ids(name)}) AND valuenum > {lo} "
                f"AND valuenum < {hi} THEN valuenum END")

    temp_expr = (
        f"CASE WHEN itemid IN ({ids('temp_f')}) AND valuenum > 50 AND valuenum < 120 "
        f"THEN (valuenum-32)/1.8 "
        f"WHEN itemid IN ({ids('temp_c')}) AND valuenum > {b['temp_c'][0]} "
        f"AND valuenum < {b['temp_c'][1]} THEN valuenum END"
    )

    simple = ["heart_rate", "resp_rate", "spo2", "sysbp", "diasbp", "map", "glucose_vit"]
    select_parts: list[str] = ["stay_id"]
    for name in simple:
        e = bounded(name)
        select_parts += [f"MIN({e}) AS {name}_min",
                         f"MAX({e}) AS {name}_max",
                         f"AVG({e}) AS {name}_mean"]
    select_parts += [f"MIN({temp_expr}) AS temp_min",
                     f"MAX({temp_expr}) AS temp_max",
                     f"AVG({temp_expr}) AS temp_mean"]

    all_ids = _flat({k: v[k] for k in simple + ["temp_f", "temp_c"]})
    id_list = ",".join(map(str, all_ids))

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE vitals AS
        SELECT {", ".join(select_parts)}
        FROM (
          SELECT ce.stay_id, ce.itemid, CAST(ce.valuenum AS DOUBLE) AS valuenum
          FROM {_read_chart(chart)} ce
          JOIN cohort c ON ce.stay_id = c.stay_id
          WHERE ce.itemid IN ({id_list})
            AND ce.valuenum IS NOT NULL
            AND ce.charttime >= c.intime
            AND ce.charttime <= c.intime + INTERVAL {win} HOUR
        )
        GROUP BY stay_id;
    """)

    # Glasgow Coma Scale total: sum the three components at each charttime,
    # then take the minimum and mean total across the window.
    ge, gv, gm = v["gcs_eye"][0], v["gcs_verbal"][0], v["gcs_motor"][0]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gcs AS
        WITH per_time AS (
          SELECT ce.stay_id, ce.charttime,
                 MAX(CASE WHEN ce.itemid={ge} THEN ce.valuenum END) AS eye,
                 MAX(CASE WHEN ce.itemid={gv} THEN ce.valuenum END) AS verbal,
                 MAX(CASE WHEN ce.itemid={gm} THEN ce.valuenum END) AS motor
          FROM {_read_chart(chart)} ce
          JOIN cohort c ON ce.stay_id = c.stay_id
          WHERE ce.itemid IN ({ge},{gv},{gm})
            AND ce.valuenum IS NOT NULL
            AND ce.charttime >= c.intime
            AND ce.charttime <= c.intime + INTERVAL {win} HOUR
          GROUP BY ce.stay_id, ce.charttime
        )
        SELECT stay_id,
               MIN(eye+verbal+motor) AS gcs_min,
               AVG(eye+verbal+motor) AS gcs_mean
        FROM per_time
        WHERE eye IS NOT NULL AND verbal IS NOT NULL AND motor IS NOT NULL
        GROUP BY stay_id;
    """)


def build_labs(cfg: dict, con: duckdb.DuckDBPyConnection) -> None:
    """Aggregate laboratory results per hospital admission into temp table `labs`."""
    labs = cfg["labs"]
    win = int(cfg["cohort"]["feature_window_hours"])
    look = int(cfg["cohort"]["lab_lookback_hours"])
    lab_path = mimic_path(cfg, "hosp", "labevents")

    select_parts: list[str] = ["c.stay_id"]
    for name, ids in labs.items():
        id_list = ",".join(map(str, ids))
        e = f"CASE WHEN le.itemid IN ({id_list}) AND le.valuenum > 0 THEN le.valuenum END"
        select_parts += [f"MIN({e}) AS {name}_min",
                         f"MAX({e}) AS {name}_max",
                         f"AVG({e}) AS {name}_mean"]

    all_ids = ",".join(map(str, _flat(labs)))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE labs AS
        SELECT {", ".join(select_parts)}
        FROM {_read_lab(lab_path)} le
        JOIN cohort c ON le.hadm_id = c.hadm_id
        WHERE le.itemid IN ({all_ids})
          AND le.valuenum IS NOT NULL
          AND le.charttime >= c.intime - INTERVAL {look} HOUR
          AND le.charttime <= c.intime + INTERVAL {win} HOUR
        GROUP BY c.stay_id;
    """)


def assemble(cfg: dict, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Join cohort + labels + labs + vitals + GCS into a single modelling frame."""
    df = con.execute("""
        SELECT
          c.stay_id, c.subject_id, c.hadm_id, c.label,
          -- demographics
          c.anchor_age AS age, c.gender,
          -- admission context
          c.admission_type, c.admission_location, c.insurance, c.first_careunit,
          -- ICU stay statistics / utilisation / comorbidity
          c.los, c.prior_icu_stays, c.prior_admissions, c.diagnosis_count,
          -- vitals
          v.* EXCLUDE (stay_id),
          g.gcs_min, g.gcs_mean,
          -- labs
          l.* EXCLUDE (stay_id)
        FROM cohort c
        LEFT JOIN vitals v ON c.stay_id = v.stay_id
        LEFT JOIN gcs    g ON c.stay_id = g.stay_id
        LEFT JOIN labs   l ON c.stay_id = l.stay_id;
    """).df()
    return df


def build_feature_frame(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Full extraction: cohort -> vitals -> labs -> assembled dataframe."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg['duckdb']['memory_limit']}'; "
                f"SET threads={cfg['duckdb']['threads']};")
    build_cohort(cfg, con)
    summary = cohort_summary(con)
    build_vitals(cfg, con)
    build_labs(cfg, con)
    df = assemble(cfg, con)
    con.close()
    return df, summary


if __name__ == "__main__":
    cfg = load_config()
    df, summary = build_feature_frame(cfg)
    print("cohort summary:", summary)
    print("feature frame:", df.shape)
    print(df.dtypes)
