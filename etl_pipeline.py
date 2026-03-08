"""
=============================================================
 PHASE 2 — Data Engineering: ETL Pipeline + SQL Database
 File   : etl_pipeline.py
 Desc   : Load raw FHIR CSV → clean → load into SQLite DB
          with a schema that mirrors real EHR data warehouses.
=============================================================

SETUP:
    pip install pandas sqlite3 (sqlite3 is built-in Python)

RUN:
    python etl_pipeline.py           (after fhir_client.py)

OUTPUT:
    data/trial_data.db
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

INPUT_CSV = "data/raw_patients.csv"
DB_PATH   = "data/trial_data.db"


# ─────────────────────────────────────────────
# SQL SCHEMA  (same structure used in EHR data warehouses)
# ─────────────────────────────────────────────
SCHEMA_SQL = """
-- ══════════════════════════════════════════
--  Clinical Trial Recruitment Database
--  Standard: mirrors eClinicalWorks / PF DWH
-- ══════════════════════════════════════════

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ─────────────────────────────────────
--  Core patient demographics
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    patient_id      TEXT PRIMARY KEY,
    given_name      TEXT,
    family_name     TEXT,
    full_name       TEXT GENERATED ALWAYS AS (given_name || ' ' || family_name) VIRTUAL,
    birth_date      TEXT,
    age             INTEGER,
    gender          TEXT CHECK(gender IN ('male','female','other','unknown','')),
    city            TEXT,
    state           TEXT,
    zip             TEXT,
    phone           TEXT,
    email           TEXT,
    language        TEXT,
    ingested_at     TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────
--  Condition flags (denormalized for fast ML feature lookup)
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS patient_conditions (
    patient_id          TEXT PRIMARY KEY REFERENCES patients(patient_id),
    condition_count     INTEGER DEFAULT 0,
    condition_codes     TEXT,   -- pipe-separated SNOMED codes
    known_conditions    TEXT,   -- pipe-separated human labels
    has_diabetes        INTEGER DEFAULT 0,
    has_hypertension    INTEGER DEFAULT 0,
    has_depression      INTEGER DEFAULT 0,
    has_asthma          INTEGER DEFAULT 0,
    has_ckd             INTEGER DEFAULT 0
);

-- ─────────────────────────────────────
--  Trial definitions
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS trials (
    trial_id        TEXT PRIMARY KEY,
    title           TEXT,
    sponsor         TEXT,
    phase           TEXT,
    min_age         INTEGER,
    max_age         INTEGER,
    gender_req      TEXT,   -- 'male','female','any'
    require_diabetes    INTEGER DEFAULT 0,
    require_hypertension INTEGER DEFAULT 0,
    require_depression  INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────
--  Eligibility scores (output of ML model)
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS eligibility_scores (
    patient_id      TEXT REFERENCES patients(patient_id),
    trial_id        TEXT REFERENCES trials(trial_id),
    match_score     REAL,
    age_match       INTEGER,
    gender_match    INTEGER,
    condition_match INTEGER,
    qualified       INTEGER GENERATED ALWAYS AS (CASE WHEN match_score >= 70 THEN 1 ELSE 0 END) VIRTUAL,
    scored_at       TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (patient_id, trial_id)
);

-- ─────────────────────────────────────
--  VIEWS for quick analysis
-- ─────────────────────────────────────

-- All qualified subjects with demographics
CREATE VIEW IF NOT EXISTS v_qualified_subjects AS
SELECT
    p.patient_id,
    p.given_name,
    p.family_name,
    p.age,
    p.gender,
    p.city,
    p.state,
    pc.known_conditions,
    pc.condition_count,
    pc.has_diabetes,
    pc.has_hypertension,
    es.match_score,
    es.trial_id
FROM patients p
JOIN patient_conditions pc ON p.patient_id = pc.patient_id
JOIN eligibility_scores es ON p.patient_id = es.patient_id
WHERE es.match_score >= 70
ORDER BY es.match_score DESC;

-- Cohort summary by gender + age band
CREATE VIEW IF NOT EXISTS v_cohort_summary AS
SELECT
    gender,
    CASE
        WHEN age < 30 THEN '18–29'
        WHEN age < 40 THEN '30–39'
        WHEN age < 50 THEN '40–49'
        WHEN age < 60 THEN '50–59'
        WHEN age < 70 THEN '60–69'
        ELSE '70+'
    END AS age_band,
    COUNT(*)            AS patient_count,
    AVG(pc.condition_count) AS avg_conditions,
    SUM(pc.has_diabetes)    AS diabetic_count
FROM patients p
JOIN patient_conditions pc USING(patient_id)
GROUP BY gender, age_band
ORDER BY gender, age_band;
"""

# Sample trials to seed the DB
SEED_TRIALS = [
    {
        "trial_id": "TRL-2024-DM2",
        "title": "Type 2 Diabetes Management Study",
        "sponsor": "Novartis",
        "phase": "Phase III",
        "min_age": 40,
        "max_age": 75,
        "gender_req": "any",
        "require_diabetes": 1,
        "require_hypertension": 0,
        "require_depression": 0,
    },
    {
        "trial_id": "TRL-2024-HTN",
        "title": "Hypertension & Cardiovascular Risk Trial",
        "sponsor": "IQVIA / Pfizer",
        "phase": "Phase II",
        "min_age": 45,
        "max_age": 80,
        "gender_req": "any",
        "require_diabetes": 0,
        "require_hypertension": 1,
        "require_depression": 0,
    },
    {
        "trial_id": "TRL-2024-DEP",
        "title": "Female Depression Treatment Study",
        "sponsor": "Flatiron Health",
        "phase": "Phase II",
        "min_age": 25,
        "max_age": 65,
        "gender_req": "female",
        "require_diabetes": 0,
        "require_hypertension": 0,
        "require_depression": 1,
    },
]


# ─────────────────────────────────────────────
# ETL FUNCTIONS
# ─────────────────────────────────────────────

def create_database(conn: sqlite3.Connection):
    print("  ✓ Creating schema...")
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean & validate the raw CSV before loading.
    This is what separates a junior from a senior Data Scientist.
    """
    print("  ✓ Cleaning data...")

    # Standardise gender values
    df["gender"] = df["gender"].str.lower().fillna("unknown")
    df.loc[~df["gender"].isin(["male", "female", "other"]), "gender"] = "unknown"

    # Fill nulls
    str_cols = ["given_name","family_name","city","state","zip","phone","email","language"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    int_cols = ["condition_count","has_diabetes","has_hypertension",
                "has_depression","has_asthma","has_ckd"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Age sanity check
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 120))]
    df["age"] = df["age"].fillna(-1).astype(int)

    # Remove duplicate patient IDs
    before = len(df)
    df = df.drop_duplicates(subset="patient_id")
    print(f"    Removed {before - len(df)} duplicate patient_ids")

    return df


def load_patients(df: pd.DataFrame, conn: sqlite3.Connection):
    """Load demographics into patients table."""
    print("  ✓ Loading patients table...")
    demo_cols = ["patient_id","given_name","family_name","birth_date","age",
                 "gender","city","state","zip","phone","email","language"]
    demo_df = df[[c for c in demo_cols if c in df.columns]]
    demo_df.to_sql("patients", conn, if_exists="replace", index=False)
    print(f"    → {len(demo_df)} rows loaded into patients")


def load_conditions(df: pd.DataFrame, conn: sqlite3.Connection):
    """Load condition flags into patient_conditions table."""
    print("  ✓ Loading patient_conditions table...")
    cond_cols = ["patient_id","condition_count","condition_codes",
                 "known_conditions","has_diabetes","has_hypertension",
                 "has_depression","has_asthma","has_ckd"]
    cond_df = df[[c for c in cond_cols if c in df.columns]]
    cond_df.to_sql("patient_conditions", conn, if_exists="replace", index=False)
    print(f"    → {len(cond_df)} rows loaded into patient_conditions")


def seed_trials(conn: sqlite3.Connection):
    """Insert sample trial definitions."""
    print("  ✓ Seeding trials table...")
    pd.DataFrame(SEED_TRIALS).to_sql("trials", conn, if_exists="replace", index=False)
    print(f"    → {len(SEED_TRIALS)} trials seeded")


def run_validation_queries(conn: sqlite3.Connection):
    """
    Run SQL validation queries — important skill to show in interviews.
    """
    print(f"\n{'─'*55}")
    print("  DATA QUALITY CHECKS (SQL)")
    print(f"{'─'*55}")

    checks = {
        "Total patients":         "SELECT COUNT(*) FROM patients",
        "Patients with age":      "SELECT COUNT(*) FROM patients WHERE age > 0",
        "Gender breakdown":       "SELECT gender, COUNT(*) FROM patients GROUP BY gender",
        "Diabetic patients":      "SELECT COUNT(*) FROM patient_conditions WHERE has_diabetes=1",
        "Hypertensive patients":  "SELECT COUNT(*) FROM patient_conditions WHERE has_hypertension=1",
        "Avg conditions/patient": "SELECT ROUND(AVG(condition_count),2) FROM patient_conditions",
        "Age band distribution":  "SELECT age_band, patient_count FROM v_cohort_summary LIMIT 10",
    }

    for label, sql in checks.items():
        try:
            result = pd.read_sql(sql, conn)
            val = result.iloc[0, 0] if len(result.columns) == 1 else "\n" + result.to_string(index=False)
            print(f"  {label:30s}: {val}")
        except Exception as e:
            print(f"  {label:30s}: ERROR — {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "═"*55)
    print("  PHASE 2 — ETL Pipeline: CSV → SQLite")
    print("═"*55 + "\n")

    # Load CSV
    if not os.path.exists(INPUT_CSV):
        print(f"  ✗ {INPUT_CSV} not found — run fhir_client.py first!")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"  ✓ Loaded {len(df)} rows from {INPUT_CSV}\n")

    # ETL
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    create_database(conn)
    df   = clean_dataframe(df)
    load_patients(df, conn)
    load_conditions(df, conn)
    seed_trials(conn)
    conn.commit()

    # Validate
    run_validation_queries(conn)

    conn.close()
    print(f"\n{'═'*55}")
    print(f"  ✅ Phase 2 complete! DB saved → {DB_PATH}")
    print(f"  Next: jupyter notebook notebooks/recruitment_model.ipynb")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
