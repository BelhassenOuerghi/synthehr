"""Build the ICU-stay cohort and the 30-day readmission label with DuckDB.

Label definition (documented in README):
  For each ICU stay with a valid outtime, label = 1 if EITHER
    (a) the same patient has a subsequent ICU stay under a DIFFERENT hospital
        admission (hadm_id) whose intime falls within 30 days after this
        stay's outtime  -- a genuine 30-day ICU readmission; OR
    (b) the patient died within 30 days of this stay's outtime (date of death
        from hosp/patients.dod, or in-hospital deathtime) -- a competing
        adverse outcome, in the spirit of Blinder-style pipelines.
  Otherwise label = 0.

At-risk exclusions:
  * ICU stays with no outtime are dropped (cannot be discharged/observed).
  * Patients who died during the ICU stay (deathtime <= outtime, or dod before
    the discharge date) are removed from the at-risk set -- they cannot be
    readmitted and their death is not a post-discharge event.
  * Subsequent stays within the SAME hadm_id are treated as intra-hospitalisation
    transfers, not readmissions, so they never trigger a positive label.
"""
from __future__ import annotations

import duckdb

from .config import load_config, mimic_path


def build_cohort(cfg: dict, con: duckdb.DuckDBPyConnection) -> None:
    """Create a DuckDB temp table `cohort` holding one row per at-risk ICU stay."""
    horizon = int(cfg["cohort"]["horizon_days"])
    icu = mimic_path(cfg, "icu", "icustays")
    pat = mimic_path(cfg, "hosp", "patients")
    adm = mimic_path(cfg, "hosp", "admissions")
    dx = mimic_path(cfg, "hosp", "diagnoses_icd")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE icu AS
        SELECT subject_id, hadm_id, stay_id, first_careunit, last_careunit,
               CAST(intime AS TIMESTAMP) AS intime,
               CAST(outtime AS TIMESTAMP) AS outtime,
               CAST(los AS DOUBLE) AS los
        FROM read_csv_auto('{icu}')
        WHERE outtime IS NOT NULL;
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pat AS
        SELECT subject_id, gender, CAST(anchor_age AS INTEGER) AS anchor_age,
               CAST(dod AS TIMESTAMP) AS dod
        FROM read_csv_auto('{pat}');
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE adm AS
        SELECT hadm_id, admission_type, admission_location, insurance,
               CAST(deathtime AS TIMESTAMP) AS deathtime,
               CAST(hospital_expire_flag AS INTEGER) AS hospital_expire_flag
        FROM read_csv_auto('{adm}');
    """)

    # Comorbidity proxy: number of coded diagnoses per hospital admission.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dx_cnt AS
        SELECT hadm_id, COUNT(*) AS diagnosis_count
        FROM read_csv_auto('{dx}')
        GROUP BY hadm_id;
    """)

    # Prior utilisation: number of earlier ICU stays and earlier distinct
    # hospital admissions for the same patient, strictly before this intime.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE prior AS
        SELECT a.stay_id,
               COUNT(DISTINCT b.stay_id) AS prior_icu_stays,
               COUNT(DISTINCT b.hadm_id) AS prior_admissions
        FROM icu a
        LEFT JOIN icu b
          ON a.subject_id = b.subject_id
         AND b.intime < a.intime
        GROUP BY a.stay_id;
    """)

    # Next qualifying ICU stay (different hadm_id, starting after this outtime).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE nxt AS
        SELECT a.stay_id,
               MIN(b.intime) AS next_intime
        FROM icu a
        LEFT JOIN icu b
          ON a.subject_id = b.subject_id
         AND b.hadm_id <> a.hadm_id
         AND b.intime > a.outtime
        GROUP BY a.stay_id;
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cohort AS
        WITH labeled AS (
          SELECT i.subject_id, i.hadm_id, i.stay_id,
                 i.first_careunit, i.last_careunit,
                 i.intime, i.outtime, i.los,
                 p.gender, p.anchor_age, p.dod,
                 a.admission_type, a.admission_location, a.insurance,
                 a.deathtime, a.hospital_expire_flag,
                 COALESCE(d.diagnosis_count, 0) AS diagnosis_count,
                 COALESCE(pr.prior_icu_stays, 0) AS prior_icu_stays,
                 COALESCE(pr.prior_admissions, 0) AS prior_admissions,
                 n.next_intime,
                 date_diff('day', i.outtime, n.next_intime) AS days_to_next,
                 date_diff('day', i.outtime, p.dod)         AS days_to_death
          FROM icu i
          JOIN pat p USING (subject_id)
          LEFT JOIN adm a  USING (hadm_id)
          LEFT JOIN dx_cnt d USING (hadm_id)
          LEFT JOIN prior pr ON i.stay_id = pr.stay_id
          LEFT JOIN nxt   n  ON i.stay_id = n.stay_id
        )
        SELECT *,
          CASE
            WHEN (days_to_next  IS NOT NULL AND days_to_next  BETWEEN 0 AND {horizon})
              OR (days_to_death IS NOT NULL AND days_to_death BETWEEN 0 AND {horizon})
            THEN 1 ELSE 0
          END AS label
        FROM labeled
        -- At-risk set: drop patients who died during / before ICU discharge.
        WHERE NOT (deathtime IS NOT NULL AND deathtime <= outtime)
          AND NOT (dod IS NOT NULL AND dod < CAST(outtime AS DATE));
    """)


def cohort_summary(con: duckdb.DuckDBPyConnection) -> dict:
    """Return aggregate (non-patient) cohort statistics for reporting."""
    row = con.execute("""
        SELECT
          COUNT(*)                                              AS n_stays,
          COUNT(DISTINCT subject_id)                            AS n_patients,
          SUM(label)                                            AS n_positive,
          SUM(CASE WHEN days_to_next  BETWEEN 0 AND 30 THEN 1 ELSE 0 END) AS n_readmit,
          SUM(CASE WHEN days_to_death BETWEEN 0 AND 30 THEN 1 ELSE 0 END) AS n_death30
        FROM cohort;
    """).fetchone()
    n_stays, n_patients, n_pos, n_readmit, n_death = row
    return {
        "n_stays": int(n_stays),
        "n_patients": int(n_patients),
        "n_positive": int(n_pos),
        "positive_rate": round(n_pos / n_stays, 4),
        "n_readmit_within_30d": int(n_readmit),
        "n_death_within_30d": int(n_death),
    }


if __name__ == "__main__":
    cfg = load_config()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg['duckdb']['memory_limit']}'; "
                f"SET threads={cfg['duckdb']['threads']};")
    build_cohort(cfg, con)
    print(cohort_summary(con))
