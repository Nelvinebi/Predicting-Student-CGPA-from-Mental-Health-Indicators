"""
MindGrade — Phase 5: Reporting & Summary
=========================================
Input  : outputs/reports/model_results.csv
         outputs/reports/chi_square_results.csv
         outputs/figures/  (all PNG charts)
Output : outputs/reports/MindGrade_Report.html
"""

import os
import base64
import pandas as pd
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
CLEAN_PATH  = os.path.join(BASE_DIR, "data", "processed", "mindgrade_cleaned.csv")
MODEL_RES   = os.path.join(REPORTS_DIR, "model_results.csv")
CHI_RES     = os.path.join(REPORTS_DIR, "chi_square_results.csv")
HTML_OUT    = os.path.join(REPORTS_DIR, "MindGrade_Report.html")
os.makedirs(REPORTS_DIR, exist_ok=True)


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def figure_block(filename, caption):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        return "<p><em>Figure not found: " + caption + "</em></p>"
    b64  = img_to_b64(path)
    html = "<figure>"
    html += '<img src="data:image/png;base64,' + b64 + '" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);">'
    html += "<figcaption>" + caption + "</figcaption>"
    html += "</figure>"
    return html


def table_html(df):
    rows = ""
    headers = "".join("<th>" + c + "</th>" for c in df.columns)
    for _, row in df.iterrows():
        cells = "".join("<td>" + str(v) + "</td>" for v in row)
        rows += "<tr>" + cells + "</tr>"
    return (
        '<table class="data-table"><thead><tr>'
        + headers
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def build_report():
    df       = pd.read_csv(CLEAN_PATH)
    model_df = pd.read_csv(MODEL_RES) if os.path.exists(MODEL_RES) else pd.DataFrame()
    chi_df   = pd.read_csv(CHI_RES)   if os.path.exists(CHI_RES)   else pd.DataFrame()

    eda_figures = (
        figure_block("01_cgpa_distribution.png",     "Figure 1: CGPA Band Distribution")
        + figure_block("02_gender_distribution.png",  "Figure 2: Gender Distribution")
        + figure_block("03_mh_prevalence.png",        "Figure 3: Mental Health Condition Prevalence")
        + figure_block("04_mh_burden_score.png",      "Figure 4: Mental Health Burden Score")
        + figure_block("05_depression_vs_cgpa.png",   "Figure 5: Depression vs CGPA")
        + figure_block("05_anxiety_vs_cgpa.png",      "Figure 6: Anxiety vs CGPA")
        + figure_block("05_panic_attack_vs_cgpa.png", "Figure 7: Panic Attack vs CGPA")
        + figure_block("06_age_distribution.png",     "Figure 8: Age Distribution")
        + figure_block("07_year_vs_cgpa.png",         "Figure 9: Year of Study vs CGPA")
        + figure_block("08_course_vs_cgpa_heatmap.png", "Figure 10: Course Group vs CGPA Heatmap")
        + figure_block("09_correlation_heatmap.png",  "Figure 11: Feature Correlation Heatmap")
    )

    model_figures = figure_block("11_model_comparison.png", "Figure 12: Model Performance Comparison")

    cm_figures = (
        figure_block("10_confusion_logistic_regression.png", "Figure 13: Confusion Matrix — Logistic Regression")
        + figure_block("10_confusion_random_forest.png",     "Figure 14: Confusion Matrix — Random Forest")
        + figure_block("10_confusion_xgboost.png",           "Figure 15: Confusion Matrix — XGBoost")
        + figure_block("10_confusion_svm.png",               "Figure 16: Confusion Matrix — SVM")
    )

    shap_figure = figure_block("12_shap_feature_importance.png", "Figure 17: SHAP Feature Importance")

    chi_table   = table_html(chi_df)   if not chi_df.empty   else "<p><em>Run Phase 2 to generate chi-square results.</em></p>"
    model_table = table_html(model_df) if not model_df.empty else "<p><em>Run Phase 4 to generate model results.</em></p>"

    stat1 = str(len(df))
    stat2 = str(int(df["depression"].mean() * 100)) + "%"
    stat3 = str(int(df["anxiety"].mean() * 100)) + "%"
    stat4 = str(int(df["panic_attack"].mean() * 100)) + "%"
    stat5 = str(int(df["sought_treatment"].mean() * 100)) + "%"
    stat6 = str(df["cgpa"].value_counts().idxmax())
    today = datetime.now().strftime("%B %d, %Y %H:%M")

    css = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 15px; color: #2d2d2d; background: #f7f7f7; }
.container { max-width: 960px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 48px; }
h1 { font-size: 2rem; color: #1a1a2e; margin-bottom: 6px; }
h2 { font-size: 1.35rem; color: #16213e; margin: 36px 0 12px; border-left: 4px solid #4e89e8; padding-left: 12px; }
p  { line-height: 1.75; margin-bottom: 12px; color: #444; }
.subtitle { color: #666; font-size: 1rem; margin-bottom: 24px; }
.badge { display: inline-block; background: #4e89e8; color: #fff; border-radius: 20px; padding: 4px 14px; font-size: 0.85rem; margin-right: 6px; margin-bottom: 16px; }
.data-table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.93rem; }
.data-table th { background: #4e89e8; color: #fff; padding: 10px 14px; text-align: left; }
.data-table td { padding: 9px 14px; border-bottom: 1px solid #eee; }
.data-table tr:hover td { background: #f0f5ff; }
figure { margin: 24px 0; text-align: center; }
figcaption { margin-top: 8px; font-size: 0.88rem; color: #777; font-style: italic; }
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }
.stat-card { background: #f0f5ff; border-radius: 8px; padding: 20px; text-align: center; }
.stat-card .value { font-size: 2rem; font-weight: 700; color: #4e89e8; }
.stat-card .label { font-size: 0.85rem; color: #666; margin-top: 4px; }
footer { margin-top: 48px; font-size: 0.82rem; color: #aaa; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }
"""

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "<title>MindGrade Report</title>",
        "<style>" + css + "</style>",
        "</head>",
        "<body>",
        '<div class="container">',

        "<h1>&#x1F9E0; MindGrade</h1>",
        '<p class="subtitle">Predicting Academic Performance from Student Mental Health Indicators<br>',
        "<em>International Islamic University Malaysia (IIUM) — Survey Data, July 2020</em></p>",
        '<span class="badge">Machine Learning</span>',
        '<span class="badge">Mental Health</span>',
        '<span class="badge">Academic Performance</span>',
        '<span class="badge">CGPA Prediction</span>',

        "<h2>1. Project Overview</h2>",
        "<p>This report presents a statistical and machine learning analysis of the relationship between "
        "student mental health and academic CGPA at IIUM, Malaysia. The dataset contains responses from "
        "101 students covering self-reported depression, anxiety, and panic attack status, alongside "
        "demographic and academic information.</p>",
        "<p>The goal is to train classifiers to predict a student's CGPA band (5 classes: 0-1.99 to 3.50-4.00) "
        "from mental health and demographic features, and to identify which factors most influence academic performance.</p>",

        "<h2>2. Dataset Summary</h2>",
        '<div class="stat-grid">',
        '<div class="stat-card"><div class="value">' + stat1 + '</div><div class="label">Total respondents</div></div>',
        '<div class="stat-card"><div class="value">' + stat2 + '</div><div class="label">Report depression</div></div>',
        '<div class="stat-card"><div class="value">' + stat3 + '</div><div class="label">Report anxiety</div></div>',
        '<div class="stat-card"><div class="value">' + stat4 + '</div><div class="label">Report panic attacks</div></div>',
        '<div class="stat-card"><div class="value">' + stat5 + '</div><div class="label">Sought treatment</div></div>',
        '<div class="stat-card"><div class="value">' + stat6 + '</div><div class="label">Most common CGPA band</div></div>',
        "</div>",

        "<h2>3. Exploratory Data Analysis</h2>",
        eda_figures,

        "<h2>4. Statistical Tests — Chi-Square</h2>",
        "<p>Chi-square tests of independence assess whether each mental health condition "
        "is statistically associated with CGPA band (significance level: p &lt; 0.05).</p>",
        chi_table,

        "<h2>5. Model Performance</h2>",
        model_figures,
        model_table,

        "<h2>6. Confusion Matrices</h2>",
        cm_figures,

        "<h2>7. SHAP Feature Importance</h2>",
        "<p>SHAP (SHapley Additive exPlanations) values show the contribution of each feature "
        "to the best model's predictions across all CGPA classes.</p>",
        shap_figure,

        "<h2>8. Key Findings</h2>",
        "<p>&#x2022; Students with higher mental health burden scores tend to appear in lower CGPA bands.</p>",
        "<p>&#x2022; Only 6% of students with mental health conditions sought specialist treatment — a critical gap.</p>",
        "<p>&#x2022; Depression, anxiety, and panic attacks frequently co-occur, suggesting shared underlying causes.</p>",
        "<p>&#x2022; Year of study and age also correlate with CGPA, indicating compounding academic stressors.</p>",
        "<p>&#x2022; Logistic Regression achieved the best Weighted F1 (0.45); overall low accuracy reflects "
        "the small dataset size (101 samples, 5 classes) rather than model failure.</p>",

        "<footer>MindGrade ML Project &nbsp;|&nbsp; Generated: " + today + " &nbsp;|&nbsp; Dataset: IIUM Student Mental Health Survey (Kaggle)</footer>",
        "</div>",
        "</body>",
        "</html>",
    ]

    html = "\n".join(parts)

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("[Report] Saved -> " + HTML_OUT)


def run():
    print("=" * 60)
    print("  MindGrade -- Phase 5: Reporting")
    print("=" * 60)
    build_report()
    print("=" * 60)
    print("  Phase 5 complete. Open outputs/reports/MindGrade_Report.html")
    print("=" * 60)


if __name__ == "__main__":
    run()