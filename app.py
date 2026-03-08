"""
=============================================================
 PHASE 4 — Interactive Dashboard
 File   : dashboard/app.py
 Desc   : Streamlit dashboard with filters, KPIs, charts,
          and CSV export. Deploy free on streamlit.io
=============================================================

SETUP:
    pip install streamlit plotly pandas

RUN locally:
    streamlit run dashboard/app.py

DEPLOY FREE:
    1. Push project to GitHub
    2. Go to share.streamlit.io
    3. Connect repo → done! Live URL in 2 minutes.
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Trial Recruiter",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — clean & professional
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-num  { font-size: 36px; font-weight: 700; color: #58a6ff; line-height: 1; }
    .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase;
                    letter-spacing: 0.1em; margin-top: 6px; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    h1, h2, h3 { color: #f0f6fc !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
DB_PATH = "data/trial_data.db"

@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DB_PATH):
        return None, None, None

    conn = sqlite3.connect(DB_PATH)

    patients = pd.read_sql("""
        SELECT p.*, pc.condition_count, pc.known_conditions,
               pc.has_diabetes, pc.has_hypertension,
               pc.has_depression, pc.has_asthma, pc.has_ckd
        FROM patients p
        JOIN patient_conditions pc ON p.patient_id = pc.patient_id
        WHERE p.age > 0
    """, conn)

    scores = pd.read_sql("SELECT * FROM eligibility_scores", conn)
    trials = pd.read_sql("SELECT * FROM trials", conn)
    conn.close()

    # Merge
    df = patients.merge(scores, on="patient_id", how="left")
    return df, scores, trials


# ─────────────────────────────────────────────
# SIDEBAR — Filters
# ─────────────────────────────────────────────
def render_sidebar(trials_df):
    st.sidebar.markdown("## 🏥 Clinical Trial Recruiter")
    st.sidebar.markdown("---")

    trial_id = st.sidebar.selectbox(
        "Select Trial",
        options=trials_df["trial_id"].tolist() if trials_df is not None else [],
        format_func=lambda x: x
    )

    st.sidebar.markdown("### Patient Filters")
    gender    = st.sidebar.selectbox("Gender", ["Any", "Female", "Male"])
    age_range = st.sidebar.slider("Age Range", 18, 90, (40, 75))
    min_score = st.sidebar.slider("Min Match Score", 0, 100, 70)

    conditions = st.sidebar.multiselect(
        "Required Conditions",
        ["Diabetes", "Hypertension", "Depression", "Asthma", "CKD"],
        default=[]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.caption("Data: FHIR R4 · smarthealthit.org sandbox")

    return trial_id, gender, age_range, min_score, conditions


# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
def apply_filters(df, gender, age_range, min_score, conditions):
    filtered = df.copy()

    if gender != "Any":
        filtered = filtered[filtered["gender"] == gender.lower()]

    filtered = filtered[
        (filtered["age"] >= age_range[0]) &
        (filtered["age"] <= age_range[1])
    ]

    if filtered["match_score"].notna().any():
        filtered = filtered[filtered["match_score"] >= min_score]

    cond_map = {
        "Diabetes"    : "has_diabetes",
        "Hypertension": "has_hypertension",
        "Depression"  : "has_depression",
        "Asthma"      : "has_asthma",
        "CKD"         : "has_ckd",
    }
    for cond in conditions:
        col = cond_map.get(cond)
        if col and col in filtered.columns:
            filtered = filtered[filtered[col] == 1]

    return filtered


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
def main():
    df, scores, trials = load_data()

    if df is None:
        st.error("⚠️  Database not found. Run `python etl_pipeline.py` and `python models/recruitment_model.py` first.")
        st.code("python fhir_client.py\npython etl_pipeline.py\npython models/recruitment_model.py")
        return

    # Sidebar
    trial_id, gender, age_range, min_score, conditions = render_sidebar(trials)

    # Filter
    filtered = apply_filters(df, gender, age_range, min_score, conditions)

    # ── Header ──
    st.markdown("# 🏥 Clinical Trial Patient Recruitment")
    if trials is not None and len(trials):
        trial_row = trials[trials["trial_id"] == trial_id]
        if len(trial_row):
            t = trial_row.iloc[0]
            st.caption(f"**{t['title']}** · {t['sponsor']} · {t['phase']} · Ages {t['min_age']}–{t['max_age']}")

    st.markdown("---")

    # ── KPI Row ──
    total     = len(filtered)
    qualified = int((filtered["match_score"] >= 70).sum()) if "match_score" in filtered.columns else 0
    avg_score = round(filtered["match_score"].mean(), 1) if "match_score" in filtered.columns and total > 0 else 0
    enroll_rt = f"{qualified / total * 100:.0f}%" if total > 0 else "—"

    col1, col2, col3, col4 = st.columns(4)
    for col, num, label in [
        (col1, total,     "Patients Screened"),
        (col2, qualified, "Qualified (≥70)"),
        (col3, avg_score, "Avg Match Score"),
        (col4, enroll_rt, "Enrollment Rate"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-num">{num}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Match Score Distribution")
        if "match_score" in filtered.columns and total > 0:
            fig = px.histogram(
                filtered, x="match_score", nbins=20,
                color_discrete_sequence=["#58a6ff"],
                template="plotly_dark",
            )
            fig.add_vline(x=70, line_dash="dash", line_color="#f85149",
                          annotation_text="Qualified threshold (70)")
            fig.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#c9d1d9", margin=dict(t=20, b=20),
                xaxis_title="Match Score", yaxis_title="# Patients"
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Age Distribution by Gender")
        if total > 0:
            fig2 = px.histogram(
                filtered, x="age", color="gender", nbins=20,
                color_discrete_map={"female": "#f778ba", "male": "#58a6ff", "unknown": "#8b949e"},
                template="plotly_dark", barmode="overlay", opacity=0.75,
            )
            fig2.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#c9d1d9", margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Conditions breakdown ──
    st.subheader("Condition Prevalence in Cohort")
    if total > 0:
        cond_data = pd.DataFrame({
            "Condition": ["Diabetes", "Hypertension", "Depression", "Asthma", "CKD"],
            "Count": [
                filtered.get("has_diabetes", pd.Series([0])).sum(),
                filtered.get("has_hypertension", pd.Series([0])).sum(),
                filtered.get("has_depression", pd.Series([0])).sum(),
                filtered.get("has_asthma", pd.Series([0])).sum(),
                filtered.get("has_ckd", pd.Series([0])).sum(),
            ]
        })
        cond_data["Prevalence %"] = (cond_data["Count"] / total * 100).round(1)

        fig3 = px.bar(
            cond_data, x="Condition", y="Prevalence %",
            color="Prevalence %", color_continuous_scale="Blues",
            template="plotly_dark", text="Prevalence %"
        )
        fig3.update_traces(texttemplate="%{text}%", textposition="outside")
        fig3.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", margin=dict(t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Patient table ──
    st.subheader(f"Patient List ({total} results)")

    display_cols = ["patient_id", "given_name", "family_name", "age", "gender",
                    "city", "state", "known_conditions", "match_score"]
    avail_cols   = [c for c in display_cols if c in filtered.columns]
    table_df     = filtered[avail_cols].sort_values("match_score", ascending=False) \
                   if "match_score" in filtered.columns else filtered[avail_cols]

    st.dataframe(
        table_df.head(200),
        use_container_width=True,
        hide_index=True,
        column_config={
            "match_score": st.column_config.ProgressColumn(
                "Match Score", min_value=0, max_value=100, format="%.1f"
            )
        }
    )

    # ── Download ──
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="⬇️  Download Qualified Patient List (CSV)",
        data=csv,
        file_name=f"qualified_patients_{trial_id}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        type="primary"
    )

    # ── SHAP chart (if exists) ──
    shap_path = "docs/shap_summary.png"
    if os.path.exists(shap_path):
        st.markdown("---")
        st.subheader("🧠 Model Explainability (SHAP)")
        st.caption("Which features matter most in predicting eligibility?")
        st.image(shap_path, use_column_width=True)


if __name__ == "__main__":
    main()
