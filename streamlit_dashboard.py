"""
MindGrade — Streamlit Interactive Dashboard
=============================================
Run:  streamlit run streamlit_dashboard.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindGrade Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_TEAL      = "#00B4D8"
C_BLUE      = "#0077B6"
C_DARK_BLUE = "#03045E"
C_GREEN     = "#52B788"
C_MINT      = "#95D5B2"
C_LIME      = "#B7E4C7"
C_WHITE     = "#F0FFFF"
C_AMBER     = "#FFB703"
C_CORAL     = "#E63946"

PALETTE     = [C_TEAL, C_GREEN, C_BLUE, C_MINT, C_AMBER, C_CORAL, C_LIME]
CGPA_ORDER  = ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"]
MH_COLS     = ["depression", "anxiety", "panic_attack"]

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Background gradient ── */
.stApp {
    background: linear-gradient(135deg,
        #03045E 0%,
        #023E8A 15%,
        #0077B6 30%,
        #0096C7 45%,
        #00B4D8 60%,
        #1B6E4F 75%,
        #52B788 90%,
        #95D5B2 100%
    );
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
}

/* ── Glassmorphism panels ── */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.20);
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #00E5FF;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.72);
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.60);
    margin-bottom: 20px;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(90deg,
        rgba(0,20,80,0.85) 0%,
        rgba(0,119,182,0.70) 50%,
        rgba(82,183,136,0.70) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.30);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.75);
    margin-top: 8px;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.18);
    border: 1px solid rgba(0,229,255,0.40);
    color: #00E5FF;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 10px 6px 0 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(3, 4, 94, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.90) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label { color: rgba(255,255,255,0.70) !important; font-size: 0.82rem !important; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.12);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.65) !important;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 20px;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0077B6, #52B788) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ── Insight boxes ── */
.insight-box {
    background: rgba(0,229,255,0.08);
    border-left: 3px solid #00E5FF;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.85);
    line-height: 1.6;
}

/* ── General text ── */
h1,h2,h3,h4,h5,h6 { color: #FFFFFF !important; font-family: 'Space Grotesk', sans-serif !important; }
p, li, div { color: rgba(255,255,255,0.85); }
.stMarkdown { color: rgba(255,255,255,0.85); }

/* ── DataFrame ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "processed", "mindgrade_cleaned.csv")
    if not os.path.exists(path):
        # fallback — same dir
        path = os.path.join(base, "mindgrade_cleaned.csv")
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_model_results():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "outputs", "reports", "model_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost", "Random Forest", "SVM"],
        "Accuracy": [0.4286, 0.3810, 0.3333, 0.1429],
        "Macro F1": [0.3800, 0.2568, 0.3295, 0.1394],
        "Weighted F1": [0.4524, 0.3744, 0.3379, 0.1111],
        "ROC-AUC": ["n/a", "n/a", "n/a", "n/a"],
    })

@st.cache_data
def load_chi_results():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "outputs", "reports", "chi_square_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "Feature": ["depression", "anxiety", "panic_attack", "mh_burden_score"],
        "Chi2": [8.9975, 3.5243, 7.3752, 8.7773],
        "p-value": [0.0612, 0.4742, 0.1173, 0.7218],
        "dof": [4, 4, 4, 12],
        "Result": ["✗ Not significant"] * 4,
    })

df         = load_data()
model_df   = load_model_results()
chi_df     = load_chi_results()

# ── Plotly template ────────────────────────────────────────────────────────────
def glass_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(family="DM Sans", color="rgba(255,255,255,0.85)", size=12),
        title_font=dict(family="Space Grotesk", size=14, color="#FFFFFF"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.25)", bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.10)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.10)"),
    )
    return fig

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2.4rem'>🧠</div>
        <div style='font-family:Space Grotesk; font-size:1.3rem; font-weight:700; color:#00E5FF;'>MindGrade</div>
        <div style='font-size:0.75rem; color:rgba(255,255,255,0.50); margin-top:4px;'>IIUM · Malaysia · 2020</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.12); margin-bottom:20px;'/>
    """, unsafe_allow_html=True)

    st.markdown("**🎛️ Filters**")

    gender_opts = ["All"] + sorted(df["gender"].unique().tolist())
    sel_gender  = st.selectbox("Gender", gender_opts)

    year_opts = ["All"] + sorted(df["year_of_study"].unique().tolist())
    sel_year  = st.selectbox("Year of Study", year_opts)

    course_opts = ["All"] + sorted(df["course"].unique().tolist())
    sel_course  = st.selectbox("Course Group", course_opts)

    cgpa_opts = ["All"] + CGPA_ORDER
    sel_cgpa  = st.selectbox("CGPA Band", cgpa_opts)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12);'/>", unsafe_allow_html=True)
    st.markdown("**📋 About**")
    st.markdown("""
    <div style='font-size:0.80rem; color:rgba(255,255,255,0.55); line-height:1.6;'>
    A 5-phase machine learning pipeline predicting CGPA from student mental health indicators.
    <br><br>
    <b style='color:rgba(255,255,255,0.75);'>n = 101 students</b><br>
    <b style='color:rgba(255,255,255,0.75);'>5 CGPA classes</b><br>
    <b style='color:rgba(255,255,255,0.75);'>4 ML models</b>
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_gender != "All":  fdf = fdf[fdf["gender"]        == sel_gender]
if sel_year   != "All":  fdf = fdf[fdf["year_of_study"] == sel_year]
if sel_course != "All":  fdf = fdf[fdf["course"]        == sel_course]
if sel_cgpa   != "All":  fdf = fdf[fdf["cgpa"]          == sel_cgpa]

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">🧠 MindGrade</div>
    <div class="hero-sub">Predicting Academic Performance from Student Mental Health Indicators · IIUM, Malaysia</div>
    <span class="hero-badge">Machine Learning</span>
    <span class="hero-badge">Mental Health</span>
    <span class="hero-badge">CGPA Prediction</span>
    <span class="hero-badge">n = {len(fdf)} students</span>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, str(len(fdf)),                                       "Students"),
    (k2, f"{fdf['depression'].mean()*100:.0f}%",              "Have Depression"),
    (k3, f"{fdf['anxiety'].mean()*100:.0f}%",                 "Have Anxiety"),
    (k4, f"{fdf['panic_attack'].mean()*100:.0f}%",            "Panic Attacks"),
    (k5, f"{fdf['sought_treatment'].mean()*100:.0f}%",        "Sought Help"),
    (k6, f"{fdf['mh_burden_score'].mean():.2f}",              "Avg MH Burden"),
]
for col, val, label in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🧠 Mental Health",
    "🎓 Academic",
    "🤖 ML Models",
    "🔬 Statistics",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Demographic and academic composition of the survey cohort</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # CGPA distribution
        cgpa_counts = fdf["cgpa"].value_counts().reindex(CGPA_ORDER).fillna(0).reset_index()
        cgpa_counts.columns = ["CGPA Band", "Count"]
        fig = px.bar(cgpa_counts, x="CGPA Band", y="Count",
                     color="CGPA Band", color_discrete_sequence=PALETTE,
                     title="CGPA Band Distribution")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    with c2:
        # Gender pie
        gen_counts = fdf["gender"].value_counts().reset_index()
        gen_counts.columns = ["Gender", "Count"]
        fig = px.pie(gen_counts, names="Gender", values="Count",
                     color_discrete_sequence=[C_TEAL, C_GREEN],
                     title="Gender Distribution",
                     hole=0.45)
        fig.update_traces(textposition="outside", textfont_size=13,
                          marker=dict(line=dict(color="rgba(0,0,0,0)", width=0)))
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Year of study
        year_counts = fdf["year_of_study"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Count"]
        year_counts["Year"] = "Year " + year_counts["Year"].astype(str)
        fig = px.bar(year_counts, x="Year", y="Count",
                     color="Year", color_discrete_sequence=PALETTE,
                     title="Year of Study Distribution")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(showlegend=False)
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    with c4:
        # Course breakdown
        course_counts = fdf["course"].value_counts().reset_index()
        course_counts.columns = ["Course", "Count"]
        fig = px.bar(course_counts, x="Count", y="Course",
                     orientation="h",
                     color="Count", color_continuous_scale=[[0, C_BLUE], [1, C_MINT]],
                     title="Students by Course Group")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    # Age distribution
    st.markdown("---")
    age_counts = fdf["age"].value_counts().sort_index().reset_index()
    age_counts.columns = ["Age", "Count"]
    fig = px.bar(age_counts, x="Age", y="Count",
                 color_discrete_sequence=[C_TEAL],
                 title="Age Distribution of Respondents")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(bargap=0.3)
    st.plotly_chart(glass_fig(fig, height=300), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MENTAL HEALTH
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Mental Health Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Prevalence, co-occurrence and burden across the student cohort</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Prevalence stacked bar
        conditions   = ["Depression", "Anxiety", "Panic Attack", "Sought Treatment"]
        yes_vals     = [fdf[c].mean()*100 for c in ["depression","anxiety","panic_attack","sought_treatment"]]
        no_vals      = [100 - v for v in yes_vals]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Yes", x=conditions, y=yes_vals,
                             marker_color=C_CORAL, marker_line_width=0))
        fig.add_trace(go.Bar(name="No",  x=conditions, y=no_vals,
                             marker_color=C_TEAL, marker_line_width=0))
        fig.update_layout(barmode="stack", title="Mental Health Condition Prevalence (%)",
                          yaxis=dict(ticksuffix="%"))
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    with c2:
        # MH burden score
        burden_counts = fdf["mh_burden_score"].value_counts().sort_index().reset_index()
        burden_counts.columns = ["Score", "Count"]
        burden_counts["Label"] = burden_counts["Score"].map(
            {0:"None (0)", 1:"One (1)", 2:"Two (2)", 3:"All three (3)"})
        fig = px.bar(burden_counts, x="Label", y="Count",
                     color="Score",
                     color_continuous_scale=[[0, C_MINT],[0.5, C_AMBER],[1, C_CORAL]],
                     title="Mental Health Burden Score")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    st.markdown("---")

    # MH condition selector vs CGPA
    st.markdown('<div class="section-title" style="font-size:1.1rem;">Mental Health vs CGPA</div>', unsafe_allow_html=True)
    sel_mh = st.radio("Select condition:", ["Depression", "Anxiety", "Panic Attack"],
                      horizontal=True)
    col_map = {"Depression": "depression", "Anxiety": "anxiety", "Panic Attack": "panic_attack"}
    col     = col_map[sel_mh]

    c3, c4 = st.columns(2)

    with c3:
        grp = fdf.groupby(["cgpa", col]).size().unstack(fill_value=0).reindex(CGPA_ORDER)
        grp.columns = ["No", "Yes"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="No",  x=grp.index, y=grp["No"],  marker_color=C_TEAL,  marker_line_width=0))
        fig.add_trace(go.Bar(name="Yes", x=grp.index, y=grp["Yes"], marker_color=C_CORAL, marker_line_width=0))
        fig.update_layout(barmode="group", title=f"{sel_mh} vs CGPA — Counts",
                          xaxis_tickangle=-15)
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    with c4:
        grp_pct = grp.div(grp.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(name="No",  x=grp_pct.index, y=grp_pct["No"],
                             marker_color=C_TEAL, marker_line_width=0))
        fig.add_trace(go.Bar(name="Yes", x=grp_pct.index, y=grp_pct["Yes"],
                             marker_color=C_CORAL, marker_line_width=0))
        fig.update_layout(barmode="stack", title=f"{sel_mh} vs CGPA — Proportions (%)",
                          yaxis=dict(ticksuffix="%"), xaxis_tickangle=-15)
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    # Burden score vs CGPA heatmap
    st.markdown("---")
    pivot = fdf.groupby(["mh_burden_score", "cgpa"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=CGPA_ORDER, fill_value=0)
    pivot.index = ["Score 0", "Score 1", "Score 2", "Score 3"]

    fig = px.imshow(pivot,
                    color_continuous_scale=[[0, "rgba(0,180,216,0.1)"],
                                            [0.5, C_TEAL],
                                            [1, C_DARK_BLUE]],
                    title="MH Burden Score × CGPA Band Heatmap",
                    text_auto=True, aspect="auto")
    fig.update_coloraxes(showscale=True)
    st.plotly_chart(glass_fig(fig, height=320), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insight:</b> Approximately 1 in 3 students reports each mental health condition,
    yet only ~6% sought specialist help — a critical treatment gap. Depression, anxiety, and panic
    attacks strongly co-occur (burden score ≥ 2 affects 28% of students), suggesting shared
    underlying stressors rather than isolated conditions.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACADEMIC
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Academic Performance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">CGPA patterns across courses, year groups, and demographics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Year vs CGPA
        pivot_yr = fdf.groupby(["year_of_study", "cgpa"]).size().unstack(fill_value=0)
        pivot_yr = pivot_yr.reindex(columns=CGPA_ORDER, fill_value=0)
        pivot_yr.index = [f"Year {y}" for y in pivot_yr.index]

        fig = px.bar(pivot_yr, barmode="group",
                     color_discrete_sequence=PALETTE,
                     title="Year of Study vs CGPA Band")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_title="Year", yaxis_title="Count",
                          legend_title="CGPA Band")
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    with c2:
        # Gender vs CGPA
        pivot_gen = fdf.groupby(["gender", "cgpa"]).size().unstack(fill_value=0)
        pivot_gen = pivot_gen.reindex(columns=CGPA_ORDER, fill_value=0)
        fig = px.bar(pivot_gen, barmode="group",
                     color_discrete_sequence=PALETTE,
                     title="Gender vs CGPA Band")
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_title="Gender", yaxis_title="Count",
                          legend_title="CGPA Band")
        st.plotly_chart(glass_fig(fig), use_container_width=True)

    # Course vs CGPA heatmap
    pivot_c = fdf.groupby(["course", "cgpa"]).size().unstack(fill_value=0)
    pivot_c = pivot_c.reindex(columns=CGPA_ORDER, fill_value=0)

    fig = px.imshow(pivot_c,
                    color_continuous_scale=[[0, "rgba(82,183,136,0.1)"],
                                            [0.5, C_GREEN],
                                            [1, C_DARK_BLUE]],
                    title="Course Group × CGPA Band Heatmap",
                    text_auto=True, aspect="auto")
    st.plotly_chart(glass_fig(fig, height=360), use_container_width=True)

    # Correlation heatmap
    st.markdown("---")
    num_cols = ["age", "year_of_study", "marital_status",
                "depression", "anxiety", "panic_attack",
                "sought_treatment", "mh_burden_score", "cgpa_label"]
    corr = fdf[num_cols].corr().round(2)

    fig = px.imshow(corr,
                    color_continuous_scale=[[0, C_CORAL], [0.5, "rgba(255,255,255,0.05)"], [1, C_TEAL]],
                    zmin=-1, zmax=1,
                    title="Feature Correlation Heatmap",
                    text_auto=True, aspect="auto")
    st.plotly_chart(glass_fig(fig, height=480), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insight:</b> Year of study shows a positive correlation with CGPA label — students
    in later years tend toward higher CGPA bands. The three mental health flags (depression, anxiety,
    panic_attack) are positively correlated with each other (r ≈ 0.3–0.5), confirming co-occurrence,
    while their individual correlations with CGPA label are weak (r &lt; 0.1).
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML MODELS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Machine Learning Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Four classifiers trained on SMOTE-balanced data — 80/20 stratified split</div>', unsafe_allow_html=True)

    # Model comparison chart
    metrics = ["Accuracy", "Macro F1", "Weighted F1"]
    mdf     = model_df.copy()
    for m in metrics:
        mdf[m] = pd.to_numeric(mdf[m], errors="coerce")

    fig = go.Figure()
    colors_m = [C_TEAL, C_GREEN, C_AMBER, C_CORAL]
    for i, row in mdf.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors_m[i % len(colors_m)],
            marker_line_width=0,
        ))
    fig.add_hline(y=0.20, line_dash="dot", line_color="rgba(255,255,255,0.30)",
                  annotation_text="Random baseline (20%)",
                  annotation_font_color="rgba(255,255,255,0.50)")
    fig.update_layout(barmode="group", title="Model Performance Comparison",
                      yaxis=dict(range=[0, 0.65], title="Score"),
                      xaxis_title="Metric")
    st.plotly_chart(glass_fig(fig, height=420), use_container_width=True)

    # Radar chart
    st.markdown("---")

    def hex_to_rgba(hex_color, alpha=0.12):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig2 = go.Figure()
    for i, row in mdf.iterrows():
        vals = [float(row[m]) if pd.notna(row[m]) else 0 for m in metrics]
        vals += [vals[0]]
        color = colors_m[i % len(colors_m)]
        fig2.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics + [metrics[0]],
            name=row["Model"],
            line_color=color,
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.12),
        ))
    fig2.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.04)",
            radialaxis=dict(visible=True, range=[0, 0.55],
                            gridcolor="rgba(255,255,255,0.12)",
                            tickfont=dict(color="rgba(255,255,255,0.60)")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.12)",
                             tickfont=dict(color="rgba(255,255,255,0.80)"))
        ),
        title="Model Comparison Radar",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)"),
        legend=dict(bgcolor="rgba(0,0,0,0.25)", bordercolor="rgba(255,255,255,0.15)", borderwidth=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Results table
    st.markdown("---")
    st.markdown('<div class="section-title" style="font-size:1.1rem;">Detailed Results Table</div>', unsafe_allow_html=True)
    st.dataframe(
        model_df.style.highlight_max(
            subset=["Accuracy", "Macro F1", "Weighted F1"],
            color="rgba(82,183,136,0.35)"
        ).format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}", "Weighted F1": "{:.4f}"}),
        use_container_width=True,
    )

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insight:</b> Logistic Regression outperforms tree-based models on this small dataset
    (Weighted F1 = 0.45). This is consistent with bias-variance tradeoff theory — on n=101 samples
    with 5 classes, simpler models generalise better. Low overall accuracy reflects class imbalance
    (90% of students in top two CGPA bands) rather than model failure.
    </div>
    """, unsafe_allow_html=True)

    # Pipeline info
    st.markdown("---")
    st.markdown('<div class="section-title" style="font-size:1.1rem;">Pipeline Configuration</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    pipe_info = [
        (c1, "Training Samples", "190 (after SMOTE)", "Original: 80"),
        (c2, "Test Samples", "21", "80/20 stratified split"),
        (c3, "Feature Count", "17", "After one-hot encoding"),
    ]
    for col, title, val, sub in pipe_info:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="font-size:1.8rem;">{val}</div>
                <div class="metric-label">{title}</div>
                <div style='font-size:0.72rem; color:rgba(255,255,255,0.40); margin-top:6px;'>{sub}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATISTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Chi-square independence tests and descriptive statistics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("#### Chi-Square Test Results")
        st.markdown('<div style="font-size:0.83rem; color:rgba(255,255,255,0.55); margin-bottom:12px;">H₀: Mental health condition is independent of CGPA band · α = 0.05</div>', unsafe_allow_html=True)

        chi_display = chi_df.copy()
        st.dataframe(chi_display, use_container_width=True, hide_index=True)

        # p-value bar
        fig = px.bar(chi_df, x="Feature", y="p-value",
                     color="p-value",
                     color_continuous_scale=[[0, C_CORAL], [0.05, C_AMBER], [1, C_TEAL]],
                     title="Chi-Square p-values by Feature")
        fig.add_hline(y=0.05, line_dash="dash", line_color=C_AMBER,
                      annotation_text="α = 0.05 threshold",
                      annotation_font_color=C_AMBER)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(glass_fig(fig, height=320), use_container_width=True)

    with c2:
        st.markdown("#### Descriptive Statistics")
        desc = fdf[["age", "year_of_study", "mh_burden_score", "cgpa_label"]].describe().round(3)
        desc.columns = ["Age", "Year", "MH Score", "CGPA Label"]
        st.dataframe(desc, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Marital status breakdown
        st.markdown("#### Marital Status")
        ms = fdf["marital_status"].value_counts().reset_index()
        ms.columns = ["Status", "Count"]
        ms["Status"] = ms["Status"].map({0: "Single", 1: "Married"})
        fig2 = px.pie(ms, names="Status", values="Count",
                      color_discrete_sequence=[C_TEAL, C_GREEN],
                      hole=0.50)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(glass_fig(fig2, height=280), use_container_width=True)

    # MH burden score distribution by CGPA
    st.markdown("---")
    fig3 = px.box(fdf, x="cgpa", y="mh_burden_score",
                  category_orders={"cgpa": CGPA_ORDER},
                  color="cgpa",
                  color_discrete_sequence=PALETTE,
                  title="Mental Health Burden Score Distribution by CGPA Band",
                  labels={"mh_burden_score": "MH Burden Score (0–3)", "cgpa": "CGPA Band"})
    fig3.update_traces(marker_line_width=0)
    fig3.update_layout(showlegend=False)
    st.plotly_chart(glass_fig(fig3, height=380), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insight:</b> Chi-square tests found no statistically significant association between
    individual mental health conditions and CGPA band (all p-values &gt; 0.05). However, this is
    likely a <b>statistical power limitation</b> of the small sample (n=101) rather than true
    absence of effect — with only 2–4 students in the lowest CGPA classes, the test cannot
    reliably detect real associations even if they exist.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding: 20px; color:rgba(255,255,255,0.35); font-size:0.78rem;'>
    🧠 MindGrade ML Project &nbsp;·&nbsp; IIUM Student Mental Health Survey (Kaggle, 2020)
    &nbsp;·&nbsp; Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
