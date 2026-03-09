# 🏥 Clinical Trial Patient Recruitment Engine
> **A production-grade Data Science pipeline** integrating Python, FHIR R4 API, SQL, Machine Learning & Streamlit — built to automate patient screening for clinical trials.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FHIR](https://img.shields.io/badge/FHIR-R4-green)](https://hl7.org/fhir/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-red?logo=streamlit)](https://share.streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Problem Statement

Manual patient screening for clinical trials is **slow, costly, and error-prone**. Clinical research teams spend 30–40% of trial timelines on recruitment alone. This project builds an automated, explainable ML pipeline that:

1. Pulls real patient data from EHR systems via **FHIR R4 API** (same standard used by eClinicalWorks & Practice Fusion)
2. Stores and queries it with a **normalized SQL schema** mirroring real EHR data warehouses
3. Scores each patient's **eligibility** using a trained ML model
4. Presents results in an **interactive Streamlit dashboard** with CSV export

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────┐
│  EHR Systems (eClinicalWorks / Practice Fusion)      │
│  ↓  FHIR R4 REST API  (OAuth2 auth)                  │
└──────────────────┬──────────────────────────────────┘
                   │
          fhir_client.py
          (Patient + Condition resources)
                   │
                   ▼
          data/raw_patients.csv
                   │
          etl_pipeline.py
          (clean → SQLite)
                   │
                   ▼
          data/trial_data.db
          ┌─────────────────┐
          │ patients         │
          │ patient_conditions│
          │ trials           │
          │ eligibility_scores│
          └────────┬────────┘
                   │
     models/recruitment_model.py
     (Random Forest + SHAP)
                   │
                   ▼
          Eligibility Scores
                   │
          dashboard/app.py
          (Streamlit)
                   │
                   ▼
     🌐 Live Dashboard (streamlit.io)
```

---

## 📊 Results

| Metric                | Value   |
|-----------------------|---------|
| Patients Screened     | 100+    |
| Model AUC (ROC)       | ~0.94   |
| Qualified Rate (≥70)  | ~35%    |
| Conditions Tracked    | 5 (SNOMED CT) |
| Screening Time Saved  | ~85% vs manual |

---

## 🛠 Tech Stack

| Layer          | Technology                              |
|----------------|-----------------------------------------|
| Data Ingestion | Python · `requests` · FHIR R4 REST API  |
| Data Storage   | SQLite · SQL schema · Pandas ETL        |
| ML Model       | Scikit-learn · Random Forest · XGBoost  |
| Explainability | SHAP (SHapley Additive exPlanations)    |
| Visualization  | Plotly · Seaborn · Matplotlib           |
| Dashboard      | Streamlit · Streamlit Cloud             |
| Standards      | FHIR R4 · HL7 · SNOMED CT · OAuth2      |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/clinical-trial-recruiter
cd clinical-trial-recruiter
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
# Phase 1: Fetch patients from FHIR API
python fhir_client.py

# Phase 2: Load into SQLite
python etl_pipeline.py

# Phase 3: Train ML model + generate SHAP plot
python models/recruitment_model.py

# Phase 4: Launch dashboard
streamlit run dashboard/app.py
```

### 3. Deploy dashboard (free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → **Live URL in 2 minutes**

---

## 📁 Project Structure

```
clinical-trial-recruiter/
├── fhir_client.py              # Phase 1: FHIR API ingestion
├── etl_pipeline.py             # Phase 2: ETL + SQL schema
├── models/
│   └── recruitment_model.py    # Phase 3: ML model + SHAP
├── dashboard/
│   └── app.py                  # Phase 4: Streamlit dashboard
├── data/                       # Generated CSV + SQLite DB
├── docs/                       # SHAP plots, report
├── requirements.txt
└── README.md
```

---

## 🔐 HIPAA & Data Privacy Note

> All patient data in this project is sourced from the **public FHIR R4 sandbox** at [smarthealthit.org](https://smarthealthit.org) — a testing environment with **no real patient data**.

In a production deployment:
- OAuth2 / SMART on FHIR authentication would be required
- All data access would require IRB approval and BAA agreements
- PHI would be de-identified per HIPAA Safe Harbor method

---

## 💡 Key Concepts Demonstrated

- **FHIR R4 API integration** — same standard used by eClinicalWorks, Epic, Cerner, Practice Fusion
- **SNOMED CT coding** — industry standard for medical conditions
- **OAuth2 authentication flow** — documented and diagrammed
- **SQL data modeling** — normalized schema with views for analytics
- **Explainable AI (SHAP)** — critical for regulated healthcare ML
- **Streamlit deployment** — live, shareable demo

---

## 👤 Author

Alexia Le 
Aspiring Clinical Data Scientist | Python · FHIR · ML  


---

*Built as a capstone project demonstrating clinical data science skills for roles at IQVIA, Novartis, Veeva Systems, Flatiron Health, and health-tech startups.*
