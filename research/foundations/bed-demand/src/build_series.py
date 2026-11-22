"""
build_series.py
---------------
Derive aggregate daily count series from MIMIC-IV v3.1 `icu/icustays.csv.gz`
with DuckDB. Two demand signals are produced (counts per calendar day only,
no patient-level fields ever leave the MIMIC directory):

  admissions : number of new ICU admissions per day  = count of icustays
               grouped by date(intime).
  census     : midnight ICU census / occupancy       = number of stays with
               date(intime) <= day < date(outtime).

The census signal is the closer analogue of the upstream target (Lee et al.
2021 forecast a hospital *census* count: total hospitalized COVID patients),
while the admissions signal is the canonical "how many new beds will be
demanded tomorrow" quantity.

Outputs land in the git-ignored `data/derived/` directory.
"""

import argparse
import os

import duckdb
import pandas as pd
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    root = cfg["mimic_root"]
    icustays = os.path.join(root, "icu", "icustays.csv.gz")
    out_dir = cfg["derived_dir"]
    os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect()

    n_null = con.execute(
        f"SELECT count(*) FROM read_csv_auto('{icustays}') WHERE outtime IS NULL"
    ).fetchone()[0]
    print(f"icustays rows with NULL outtime (excluded from census): {n_null}")

    admissions = con.execute(f"""
        WITH days AS (
          SELECT unnest(generate_series(DATE '{cfg["window_start"]}',
                                        DATE '{cfg["window_end"]}',
                                        INTERVAL 1 DAY))::DATE AS day
        ),
        adm AS (
          SELECT intime::DATE AS day, count(*) AS n
          FROM read_csv_auto('{icustays}')
          GROUP BY 1
        )
        SELECT d.day, coalesce(a.n, 0) AS count
        FROM days d LEFT JOIN adm a USING (day)
        ORDER BY d.day
    """).df()

    census = con.execute(f"""
        WITH days AS (
          SELECT unnest(generate_series(DATE '{cfg["window_start"]}',
                                        DATE '{cfg["window_end"]}',
                                        INTERVAL 1 DAY))::DATE AS day
        ),
        stays AS (
          SELECT intime::DATE AS ind, outtime::DATE AS outd
          FROM read_csv_auto('{icustays}')
          WHERE outtime IS NOT NULL
        )
        SELECT d.day, count(s.ind) AS count
        FROM days d LEFT JOIN stays s ON s.ind <= d.day AND s.outd > d.day
        GROUP BY 1 ORDER BY 1
    """).df()

    for name, df in [("admissions", admissions), ("census", census)]:
        path = os.path.join(out_dir, f"{name}_daily.csv")
        df.to_csv(path, index=False)
        y = df["count"].to_numpy(dtype=float)
        print(f"{name}: N={len(y)} days [{df['day'].iloc[0]} .. {df['day'].iloc[-1]}] "
              f"mean={y.mean():.3f} var={y.var(ddof=1):.3f} "
              f"Var/Mean={y.var(ddof=1)/y.mean():.3f} zeros={(y == 0).mean():.2%} "
              f"-> {path}")

    return admissions, census


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    build(args.config)
