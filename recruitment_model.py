"""
=============================================================
 PHASE 3 — Machine Learning: Patient Eligibility Scoring
 File   : models/recruitment_model.py
 Desc   : Train Random Forest to predict eligibility score.
          Uses SHAP for explainability (KEY for DS interviews).
=============================================================

SETUP:
    pip install scikit-learn shap matplotlib seaborn joblib

RUN:
    python models/recruitment_model.py

OUTPUT:
    models/eligibility_model.pkl
    data/trial_data.db  (eligibility_scores table populated)
    docs/shap_summary.png
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DB_PATH    = "data/trial_data.db"
MODEL_PATH = "models/eligibility_model.pkl"
TRIAL_ID   = "TRL-2024-DM2"   # Which trial to train for


# ─────────────────────────────────────────────
# STEP 1: Load data from SQLite
# ─────────────────────────────────────────────
def load_training_data(trial_id: str) -> pd.DataFrame:
    """
    Load patient + condition data from the database.
    Joins patients ↔ patient_conditions.
    """
    print("  ✓ Loading training data from SQLite...")
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT
            p.patient_id,
            p.age,
            p.gender,
            pc.condition_count,
            pc.has_diabetes,
            pc.has_hypertension,
            pc.has_depression,
            pc.has_asthma,
            pc.has_ckd
        FROM patients p
        JOIN patient_conditions pc ON p.patient_id = pc.patient_id
        WHERE p.age > 0
    """, conn)

    # Fetch trial requirements
    trial = pd.read_sql(
        f"SELECT * FROM trials WHERE trial_id = '{trial_id}'", conn
    ).iloc[0]

    conn.close()
    print(f"  ✓ Loaded {len(df)} patients  |  Trial: {trial['title']}\n")
    return df, trial


# ─────────────────────────────────────────────
# STEP 2: Feature engineering
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, trial: pd.Series):
    """
    Create ML features from patient + trial requirements.
    This mirrors what a clinical data scientist does at IQVIA/Novartis.
    """
    X = df.copy()

    # Encode gender
    X["gender_female"] = (X["gender"] == "female").astype(int)
    X["gender_male"]   = (X["gender"] == "male").astype(int)

    # Age features
    X["age_sq"]        = X["age"] ** 2
    X["age_in_range"]  = (
        (X["age"] >= trial["min_age"]) & (X["age"] <= trial["max_age"])
    ).astype(int)

    # Comorbidity burden
    cond_cols = ["has_diabetes","has_hypertension","has_depression",
                 "has_asthma","has_ckd"]
    X["comorbidity_score"] = X[cond_cols].sum(axis=1)

    # Drop non-feature cols
    feature_cols = [
        "age", "age_sq", "age_in_range",
        "gender_female", "gender_male",
        "condition_count", "comorbidity_score",
        "has_diabetes", "has_hypertension",
        "has_depression", "has_asthma", "has_ckd",
    ]
    return X[feature_cols], feature_cols


def create_labels(df: pd.DataFrame, trial: pd.Series) -> pd.Series:
    """
    Rule-based label: 1 = eligible, 0 = not eligible.
    In a real project this comes from a clinical team's review.
    Here we simulate it from trial criteria.
    """
    eligible = (
        (df["age"] >= trial["min_age"]) &
        (df["age"] <= trial["max_age"]) &
        (
            (trial["gender_req"] == "any") |
            (df["gender"] == trial["gender_req"])
        ) &
        (
            (trial["require_diabetes"] == 0) | (df["has_diabetes"] == 1)
        ) &
        (
            (trial["require_hypertension"] == 0) | (df["has_hypertension"] == 1)
        )
    )
    return eligible.astype(int)


# ─────────────────────────────────────────────
# STEP 3: Train & evaluate models
# ─────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    """
    Compare 3 models — shows Data Scientist rigor in interviews.
    """
    print(f"{'─'*55}")
    print("  STEP 3 — Model Training & Comparison")
    print(f"{'─'*55}\n")

    models = {
        "Random Forest"      : RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, random_state=42))
        ]),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        test_auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        preds       = model.predict(X_test)
        results[name] = {
            "model"    : model,
            "cv_auc"   : cv_scores.mean(),
            "test_auc" : test_auc,
            "preds"    : preds,
        }
        print(f"  {name:<25} CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  |  Test AUC: {test_auc:.3f}")

    # Pick best model
    best_name = max(results, key=lambda k: results[k]["test_auc"])
    print(f"\n  ✓ Best model: {best_name}  (AUC = {results[best_name]['test_auc']:.3f})\n")
    return results[best_name]["model"], best_name, results


# ─────────────────────────────────────────────
# STEP 4: SHAP explainability
# ─────────────────────────────────────────────
def explain_model(model, X_test, feature_cols: list):
    """
    SHAP = SHapley Additive exPlanations.
    This is the #1 skill that impresses clinical DS interviewers.
    It shows WHY the model made each prediction — critical for HIPAA contexts.
    """
    print(f"{'─'*55}")
    print("  STEP 4 — SHAP Explainability")
    print(f"{'─'*55}\n")

    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    Xt  = model.named_steps["scaler"].transform(X_test) if hasattr(model, "named_steps") else X_test.values

    explainer   = shap.TreeExplainer(clf) if hasattr(clf, "estimators_") else shap.LinearExplainer(clf, Xt)
    shap_values = explainer.shap_values(Xt)

    # For binary classifiers, take class=1 SHAP values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Summary plot
    os.makedirs("docs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test if not hasattr(model, "named_steps") else pd.DataFrame(Xt, columns=feature_cols),
                      feature_names=feature_cols, show=False, plot_type="bar")
    plt.title("Feature Importance — SHAP Values\n(Clinical Trial Eligibility Model)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig("docs/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ SHAP summary plot saved → docs/shap_summary.png")

    # Mean absolute SHAP values
    mean_shap = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(sv).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n  Top 5 most important features:")
    for _, row in mean_shap.head(5).iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 40)
        print(f"    {row['feature']:<25} {bar} {row['mean_abs_shap']:.4f}")

    return mean_shap


# ─────────────────────────────────────────────
# STEP 5: Score all patients & write to DB
# ─────────────────────────────────────────────
def score_all_patients(model, df_all: pd.DataFrame, trial: pd.Series,
                       feature_cols: list, trial_id: str):
    """
    Run the trained model on every patient and write scores to SQLite.
    """
    print(f"\n{'─'*55}")
    print("  STEP 5 — Scoring all patients → eligibility_scores")
    print(f"{'─'*55}")

    X_all, _ = build_features(df_all, trial)
    proba     = model.predict_proba(X_all)[:, 1]

    scores_df = pd.DataFrame({
        "patient_id"     : df_all["patient_id"].values,
        "trial_id"       : trial_id,
        "match_score"    : (proba * 100).round(1),
        "age_match"      : ((df_all["age"] >= trial["min_age"]) & (df_all["age"] <= trial["max_age"])).astype(int).values,
        "gender_match"   : ((trial["gender_req"] == "any") | (df_all["gender"] == trial["gender_req"])).astype(int).values,
        "condition_match": df_all["has_diabetes"].values if trial["require_diabetes"] else np.ones(len(df_all), dtype=int),
    })

    conn = sqlite3.connect(DB_PATH)
    scores_df.to_sql("eligibility_scores", conn, if_exists="replace", index=False)

    qualified = (scores_df["match_score"] >= 70).sum()
    print(f"  ✓ Scored {len(scores_df)} patients")
    print(f"  ✓ Qualified (≥70): {qualified} ({qualified/len(scores_df)*100:.1f}%)")
    print(f"  ✓ Avg score: {scores_df['match_score'].mean():.1f}")

    # Show top 5 candidates
    top5 = scores_df.nlargest(5, "match_score")[["patient_id","match_score","age_match","condition_match"]]
    print(f"\n  Top 5 candidates:\n{top5.to_string(index=False)}")

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "═"*55)
    print("  PHASE 3 — ML Eligibility Scoring Model")
    print("═"*55 + "\n")

    os.makedirs("models", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # Load
    df, trial = load_training_data(TRIAL_ID)

    # Features + labels
    X, feature_cols = build_features(df, trial)
    y               = create_labels(df, trial)
    print(f"  Class balance: {y.value_counts().to_dict()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train
    best_model, best_name, all_results = train_models(X_train, X_test, y_train, y_test)

    # Classification report
    print(f"\n  Classification Report — {best_name}:")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=["Not Eligible","Eligible"]))

    # SHAP
    explain_model(best_model, X_test, feature_cols)

    # Score all patients
    score_all_patients(best_model, df, trial, feature_cols, TRIAL_ID)

    # Save model
    joblib.dump({"model": best_model, "features": feature_cols, "trial": trial.to_dict()}, MODEL_PATH)
    print(f"\n  ✓ Model saved → {MODEL_PATH}")

    print(f"\n{'═'*55}")
    print("  ✅ Phase 3 complete!")
    print("  Next: streamlit run dashboard/app.py")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
