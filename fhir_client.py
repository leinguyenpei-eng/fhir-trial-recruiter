"""
=============================================================
 PHASE 1 — FHIR API Data Ingestion
 File   : fhir_client.py
 Author : <your name>
 Desc   : Connect to FHIR R4 sandbox (same API structure as
          eClinicalWorks & Practice Fusion) and pull patient
          demographics + conditions into a CSV file.
=============================================================

SETUP (run once in terminal):
    pip install requests pandas tqdm

RUN:
    python fhir_client.py

OUTPUT:
    data/raw_patients.csv
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, date
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FHIR_BASE   = "https://r4.smarthealthit.org"   # Public sandbox — same FHIR R4 API structure as eCW / PF
OUTPUT_FILE = "data/raw_patients.csv"
MAX_PATIENTS = 100    # Increase to 500+ for real project
BATCH_SIZE   = 50     # FHIR _count per request

HEADERS = {
    "Accept": "application/fhir+json",
    "Content-Type": "application/fhir+json",
}

# ─────────────────────────────────────────────
# SNOMED CT condition codes (same codes used in eClinicalWorks)
# ─────────────────────────────────────────────
SNOMED_LABELS = {
    "44054006":  "Type 2 Diabetes",
    "73211009":  "Diabetes Mellitus",
    "38341003":  "Hypertension",
    "195967001": "Asthma",
    "35489007":  "Depression",
    "709044004": "Chronic Kidney Disease",
    "22298006":  "Myocardial Infarction",
    "414545008": "Ischemic Heart Disease",
    "40055000":  "Chronic Sinusitis",
    "370143000": "Major Depressive Disorder",
}


# ─────────────────────────────────────────────
# HELPER: Safe GET with retry
# ─────────────────────────────────────────────
def fhir_get(path: str, params: dict = None, retries: int = 3) -> dict:
    """
    Call a FHIR endpoint. Returns parsed JSON or raises on failure.
    Equivalent to what fhirpy does under the hood — but transparent.
    """
    url = f"{FHIR_BASE}/{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # exponential back-off
            else:
                print(f"  ✗ Failed after {retries} attempts: {url} — {e}")
                return {}


# ─────────────────────────────────────────────
# STEP 1: Fetch patients
# ─────────────────────────────────────────────
def fetch_patients(max_count: int = MAX_PATIENTS) -> list[dict]:
    """
    Paginate through the FHIR Patient endpoint.
    Returns a list of raw FHIR Patient resources.

    FHIR API call (same in eClinicalWorks):
        GET /Patient?_count=50&_format=json
    """
    print(f"\n{'─'*55}")
    print("  STEP 1 — Fetching patients from FHIR R4 sandbox")
    print(f"  Endpoint : {FHIR_BASE}/Patient")
    print(f"  Target   : {max_count} patients")
    print(f"{'─'*55}")

    all_patients = []
    next_url = None
    params = {"_count": BATCH_SIZE, "_format": "json"}

    while len(all_patients) < max_count:
        if next_url:
            bundle = requests.get(next_url, headers=HEADERS, timeout=15).json()
        else:
            bundle = fhir_get("Patient", params)

        entries = bundle.get("entry", [])
        if not entries:
            break

        all_patients.extend([e["resource"] for e in entries])
        print(f"  ✓ Fetched {len(all_patients)} patients so far...")

        # Follow pagination (FHIR "next" link)
        links = {lnk["relation"]: lnk["url"] for lnk in bundle.get("link", [])}
        next_url = links.get("next")
        if not next_url:
            break

    print(f"  ✓ Total patients fetched: {len(all_patients)}\n")
    return all_patients[:max_count]


# ─────────────────────────────────────────────
# STEP 2: Parse patient demographics
# ─────────────────────────────────────────────
def parse_patient(resource: dict) -> dict:
    """
    Extract clean demographics from raw FHIR Patient resource.

    FHIR Patient fields mirror what eClinicalWorks / Practice Fusion stores:
    - name, birthDate, gender, address, telecom
    """
    def get_name(name_list):
        if not name_list:
            return "", ""
        n = name_list[0]
        given  = " ".join(n.get("given", []))
        family = n.get("family", "")
        return given, family

    def calc_age(birth_str):
        if not birth_str:
            return None
        bd  = datetime.strptime(birth_str[:10], "%Y-%m-%d").date()
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))

    def get_address(addr_list):
        if not addr_list:
            return "", "", ""
        a = addr_list[0]
        city  = a.get("city", "")
        state = a.get("state", "")
        zip_  = a.get("postalCode", "")
        return city, state, zip_

    given, family = get_name(resource.get("name"))
    city, state, zip_ = get_address(resource.get("address"))

    return {
        "patient_id" : resource.get("id", ""),
        "given_name" : given,
        "family_name": family,
        "birth_date" : resource.get("birthDate", ""),
        "age"        : calc_age(resource.get("birthDate")),
        "gender"     : resource.get("gender", ""),
        "city"       : city,
        "state"      : state,
        "zip"        : zip_,
        "phone"      : next((t["value"] for t in resource.get("telecom", [])
                             if t.get("system") == "phone"), ""),
        "email"      : next((t["value"] for t in resource.get("telecom", [])
                             if t.get("system") == "email"), ""),
        "language"   : resource.get("communication", [{}])[0]
                        .get("language", {}).get("coding", [{}])[0]
                        .get("code", "") if resource.get("communication") else "",
    }


# ─────────────────────────────────────────────
# STEP 3: Fetch conditions per patient
# ─────────────────────────────────────────────
def fetch_conditions(patient_id: str) -> list[dict]:
    """
    Fetch all conditions for a patient.

    FHIR API call (same in eClinicalWorks):
        GET /Condition?patient={id}&_count=20&_format=json
    """
    bundle = fhir_get("Condition", {
        "patient": patient_id,
        "_count": 20,
        "_format": "json"
    })
    conditions = []
    for entry in bundle.get("entry", []):
        res    = entry.get("resource", {})
        coding = res.get("code", {}).get("coding", [{}])[0]
        code   = coding.get("code", "")
        conditions.append({
            "code"   : code,
            "display": coding.get("display") or res.get("code", {}).get("text", ""),
            "label"  : SNOMED_LABELS.get(code, ""),
            "onset"  : res.get("onsetDateTime", "")[:10] if res.get("onsetDateTime") else "",
            "status" : res.get("clinicalStatus", {})
                         .get("coding", [{}])[0].get("code", ""),
        })
    return conditions


# ─────────────────────────────────────────────
# STEP 4: Enrich patients with conditions
# ─────────────────────────────────────────────
def enrich_patients(patients_raw: list[dict]) -> pd.DataFrame:
    """
    For each patient: parse demographics + fetch all conditions.
    Produces one row per patient with aggregated condition data.
    """
    print(f"{'─'*55}")
    print(f"  STEP 2 — Parsing demographics + fetching conditions")
    print(f"{'─'*55}")

    records = []
    for raw in tqdm(patients_raw, desc="  Processing", unit="patient"):
        p = parse_patient(raw)

        # Fetch conditions from FHIR API
        conditions = fetch_conditions(p["patient_id"])
        time.sleep(0.05)   # gentle rate-limiting — good practice

        # Aggregate condition info into flat columns
        codes    = [c["code"]    for c in conditions if c["code"]]
        displays = [c["display"] for c in conditions if c["display"]]
        knowns   = [SNOMED_LABELS[c] for c in codes if c in SNOMED_LABELS]

        p["condition_count"]     = len(conditions)
        p["condition_codes"]     = "|".join(codes)
        p["condition_displays"]  = "|".join(displays)
        p["known_conditions"]    = "|".join(knowns)
        p["has_diabetes"]        = int(any(c in codes for c in ["44054006", "73211009"]))
        p["has_hypertension"]    = int("38341003" in codes)
        p["has_depression"]      = int(any(c in codes for c in ["35489007", "370143000"]))
        p["has_asthma"]          = int("195967001" in codes)
        p["has_ckd"]             = int("709044004" in codes)

        records.append(p)

    df = pd.DataFrame(records)
    print(f"\n  ✓ Enriched {len(df)} patients with condition data")
    return df


# ─────────────────────────────────────────────
# STEP 5: Save to CSV
# ─────────────────────────────────────────────
def save_output(df: pd.DataFrame, filepath: str = OUTPUT_FILE):
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\n{'─'*55}")
    print(f"  ✓ Saved {len(df)} rows → {filepath}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "═"*55)
    print("  FHIR R4 Patient Data Ingestion Pipeline")
    print("  Sandbox: smarthealthit.org (eCW-compatible)")
    print("═"*55)

    patients_raw = fetch_patients(max_count=MAX_PATIENTS)
    df           = enrich_patients(patients_raw)
    save_output(df)

    # Quick summary
    print("\n  📊 DATASET SUMMARY")
    print(f"  Total patients     : {len(df)}")
    print(f"  With Diabetes      : {df['has_diabetes'].sum()}")
    print(f"  With Hypertension  : {df['has_hypertension'].sum()}")
    print(f"  With Depression    : {df['has_depression'].sum()}")
    print(f"  With Asthma        : {df['has_asthma'].sum()}")
    print(f"  Avg conditions/pt  : {df['condition_count'].mean():.1f}")
    print(f"  Gender breakdown   :\n{df['gender'].value_counts().to_string()}")
    print(f"\n  ✅ Phase 1 complete! Next: python etl_pipeline.py\n")


if __name__ == "__main__":
    main()
