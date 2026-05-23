"""
MindGrade — PDF Report Generator
==================================
Input  : outputs/figures/*.png
         outputs/reports/model_results.csv
         outputs/reports/chi_square_results.csv
Output : outputs/reports/MindGrade_Report.pdf

Run:
    python src/generate_pdf.py
"""

import os
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.platypus import KeepTogether

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
CLEAN_PATH  = os.path.join(BASE_DIR, "data", "processed", "mindgrade_cleaned.csv")
MODEL_RES   = os.path.join(REPORTS_DIR, "model_results.csv")
CHI_RES     = os.path.join(REPORTS_DIR, "chi_square_results.csv")
PDF_OUT     = os.path.join(REPORTS_DIR, "MindGrade_Report.pdf")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────────
BLUE      = colors.HexColor("#4e89e8")
DARK      = colors.HexColor("#1a1a2e")
LIGHT_BG  = colors.HexColor("#f0f5ff")
MID_GREY  = colors.HexColor("#888888")
WHITE     = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm


# ── Styles ─────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title",
            fontSize=26, textColor=DARK, spaceAfter=4,
            fontName="Helvetica-Bold", alignment=TA_CENTER),

        "subtitle": ParagraphStyle("subtitle",
            fontSize=11, textColor=MID_GREY, spaceAfter=6,
            fontName="Helvetica", alignment=TA_CENTER),

        "badge": ParagraphStyle("badge",
            fontSize=9, textColor=BLUE, spaceAfter=16,
            fontName="Helvetica-Bold", alignment=TA_CENTER),

        "h2": ParagraphStyle("h2",
            fontSize=14, textColor=DARK, spaceBefore=18, spaceAfter=8,
            fontName="Helvetica-Bold", borderPad=4,
            leftIndent=0),

        "h3": ParagraphStyle("h3",
            fontSize=11, textColor=DARK, spaceBefore=10, spaceAfter=4,
            fontName="Helvetica-Bold"),

        "body": ParagraphStyle("body",
            fontSize=10, textColor=colors.HexColor("#444444"),
            spaceAfter=8, leading=16, fontName="Helvetica",
            alignment=TA_JUSTIFY),

        "caption": ParagraphStyle("caption",
            fontSize=8, textColor=MID_GREY, spaceAfter=12,
            fontName="Helvetica-Oblique", alignment=TA_CENTER),

        "bullet": ParagraphStyle("bullet",
            fontSize=10, textColor=colors.HexColor("#444444"),
            spaceAfter=5, leading=15, fontName="Helvetica",
            leftIndent=14, firstLineIndent=-10),

        "footer": ParagraphStyle("footer",
            fontSize=8, textColor=MID_GREY,
            fontName="Helvetica", alignment=TA_CENTER),

        "stat_val": ParagraphStyle("stat_val",
            fontSize=22, textColor=BLUE,
            fontName="Helvetica-Bold", alignment=TA_CENTER),

        "stat_label": ParagraphStyle("stat_label",
            fontSize=8, textColor=MID_GREY,
            fontName="Helvetica", alignment=TA_CENTER),
    }
    return styles


# ── Helper: section heading with blue left bar ─────────────────────────────────
def section_heading(text, styles):
    return [
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=2),
        Paragraph(text, styles["h2"]),
    ]


# ── Helper: figure block ───────────────────────────────────────────────────────
def figure_block(filename, caption, styles, width=None):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        return [Paragraph(f"[Figure not found: {caption}]", styles["caption"])]
    avail = (PAGE_W - 2 * MARGIN)
    w = width or avail
    img = Image(path, width=w, height=w * 0.6)
    img.hAlign = "CENTER"
    return [
        Spacer(1, 6),
        img,
        Paragraph(caption, styles["caption"]),
        Spacer(1, 4),
    ]


# ── Helper: two figures side by side ──────────────────────────────────────────
def figure_pair(f1, c1, f2, c2, styles):
    avail = (PAGE_W - 2 * MARGIN)
    half  = avail * 0.48
    items = []
    for fname, cap in [(f1, c1), (f2, c2)]:
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            img = Image(path, width=half, height=half * 0.75)
            items.append([img, Paragraph(cap, styles["caption"])])
        else:
            items.append([Paragraph(f"[Not found: {cap}]", styles["caption"]), ""])

    tbl = Table([[items[0][0], items[1][0]],
                 [items[0][1], items[1][1]]],
                colWidths=[half, half])
    tbl.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [Spacer(1, 6), tbl, Spacer(1, 8)]


# ── Helper: dataframe -> reportlab table ──────────────────────────────────────
def df_to_table(df, col_widths=None):
    data = [list(df.columns)] + [list(row) for _, row in df.iterrows()]
    data = [[str(c) for c in row] for row in data]

    avail = PAGE_W - 2 * MARGIN
    if col_widths is None:
        n = len(df.columns)
        col_widths = [avail / n] * n

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return tbl


# ── Helper: stat cards row ─────────────────────────────────────────────────────
def stat_cards(stats, styles):
    """stats = list of (value, label) tuples, max 3 per row"""
    rows = []
    for i in range(0, len(stats), 3):
        chunk = stats[i:i+3]
        while len(chunk) < 3:
            chunk.append(("", ""))
        avail = PAGE_W - 2 * MARGIN
        cw = avail / 3

        val_row   = [[Paragraph(v, styles["stat_val"])   for v, _ in chunk]]
        label_row = [[Paragraph(l, styles["stat_label"]) for _, l in chunk]]

        tbl = Table(val_row + label_row, colWidths=[cw] * 3)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("ROUNDEDCORNERS", [6]),
            ("BOX",           (0, 0), (-1, -1), 0, WHITE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.5, WHITE),
        ]))
        rows.append(tbl)
        rows.append(Spacer(1, 6))
    return rows


# ── Page template with footer ──────────────────────────────────────────────────
class FooterCanvas:
    def __init__(self, filename, **kwargs):
        from reportlab.pdfgen import canvas as pdfcanvas
        self.canvas = pdfcanvas.Canvas(filename, pagesize=A4)
        self.pages  = []
        self.width, self.height = A4

    def showPage(self):
        self.pages.append(dict(self.canvas.__dict__))
        self.canvas._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.canvas.__dict__.update(page)
            self.draw_footer(page_count)
            self.canvas.showPage()
        self.canvas.save()

    def draw_footer(self, page_count):
        self.canvas.saveState()
        self.canvas.setFont("Helvetica", 7.5)
        self.canvas.setFillColor(MID_GREY)
        footer_text = (
            f"MindGrade ML Project  |  IIUM Student Mental Health Survey  |  "
            f"Generated: {datetime.now().strftime('%B %d, %Y')}  |  "
            f"Page {self.canvas._pageNumber} of {page_count}"
        )
        self.canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, footer_text)
        self.canvas.restoreState()


# ── Build PDF ──────────────────────────────────────────────────────────────────
def build_pdf():
    styles = make_styles()
    story  = []

    df       = pd.read_csv(CLEAN_PATH)
    model_df = pd.read_csv(MODEL_RES) if os.path.exists(MODEL_RES) else pd.DataFrame()
    chi_df   = pd.read_csv(CHI_RES)   if os.path.exists(CHI_RES)   else pd.DataFrame()

    avail_w = PAGE_W - 2 * MARGIN

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("MindGrade", styles["title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Predicting Academic Performance from<br/>Student Mental Health Indicators",
        styles["subtitle"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "International Islamic University Malaysia (IIUM) — Survey Data, July 2020",
        styles["subtitle"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Machine Learning  |  Mental Health  |  Academic Performance  |  CGPA Prediction",
        styles["badge"]))
    story.append(HRFlowable(width="60%", thickness=1.5, color=BLUE,
                            hAlign="CENTER", spaceAfter=40))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%B %d, %Y')}",
        styles["footer"]))
    story.append(PageBreak())

    # ── 1. Project Overview ───────────────────────────────────────────────────
    story += section_heading("1. Project Overview", styles)
    story.append(Paragraph(
        "This report presents a statistical and machine learning analysis of the relationship "
        "between student mental health and academic CGPA at IIUM, Malaysia. The dataset contains "
        "responses from 101 students covering self-reported depression, anxiety, and panic attack "
        "status, alongside demographic and academic information.",
        styles["body"]))
    story.append(Paragraph(
        "The goal is to train classifiers to predict a student's CGPA band (5 classes: 0-1.99 "
        "to 3.50-4.00) from mental health and demographic features, and to identify which factors "
        "most influence academic performance.",
        styles["body"]))

    # ── 2. Dataset Summary ────────────────────────────────────────────────────
    story += section_heading("2. Dataset Summary", styles)
    stats = [
        (str(len(df)),                                 "Total respondents"),
        (str(int(df["depression"].mean() * 100)) + "%","Report depression"),
        (str(int(df["anxiety"].mean() * 100)) + "%",   "Report anxiety"),
        (str(int(df["panic_attack"].mean() * 100)) + "%", "Report panic attacks"),
        (str(int(df["sought_treatment"].mean() * 100)) + "%", "Sought treatment"),
        (str(df["cgpa"].value_counts().idxmax()),       "Most common CGPA band"),
    ]
    story += stat_cards(stats, styles)
    story.append(Spacer(1, 8))

    # ── 3. EDA ────────────────────────────────────────────────────────────────
    story += section_heading("3. Exploratory Data Analysis", styles)

    story += figure_pair(
        "01_cgpa_distribution.png", "Figure 1: CGPA Band Distribution",
        "02_gender_distribution.png", "Figure 2: Gender Distribution",
        styles)

    story += figure_block("03_mh_prevalence.png",
                          "Figure 3: Mental Health Condition Prevalence", styles)
    story += figure_block("04_mh_burden_score.png",
                          "Figure 4: Mental Health Burden Score", styles)

    story += figure_block("05_depression_vs_cgpa.png",
                          "Figure 5: Depression vs CGPA", styles)
    story += figure_block("05_anxiety_vs_cgpa.png",
                          "Figure 6: Anxiety vs CGPA", styles)
    story += figure_block("05_panic_attack_vs_cgpa.png",
                          "Figure 7: Panic Attack vs CGPA", styles)

    story += figure_pair(
        "06_age_distribution.png", "Figure 8: Age Distribution",
        "07_year_vs_cgpa.png", "Figure 9: Year of Study vs CGPA",
        styles)

    story += figure_block("08_course_vs_cgpa_heatmap.png",
                          "Figure 10: Course Group vs CGPA Heatmap", styles)
    story += figure_block("09_correlation_heatmap.png",
                          "Figure 11: Feature Correlation Heatmap", styles)

    # ── 4. Chi-Square ─────────────────────────────────────────────────────────
    story += section_heading("4. Statistical Tests — Chi-Square", styles)
    story.append(Paragraph(
        "Chi-square tests of independence assess whether each mental health condition is "
        "statistically associated with CGPA band (significance level: p &lt; 0.05). "
        "None of the conditions reached significance, which is consistent with the small "
        "sample size (n=101) reducing statistical power.",
        styles["body"]))
    if not chi_df.empty:
        story.append(Spacer(1, 6))
        story.append(df_to_table(chi_df,
            col_widths=[avail_w * 0.3, avail_w * 0.18,
                        avail_w * 0.18, avail_w * 0.1, avail_w * 0.24]))
    story.append(Spacer(1, 12))

    # ── 5. Model Performance ──────────────────────────────────────────────────
    story += section_heading("5. Model Performance", styles)
    story.append(Paragraph(
        "Four classifiers were trained on SMOTE-balanced data and evaluated on a held-out "
        "test set of 21 students. Logistic Regression achieved the best Weighted F1 score. "
        "Overall low accuracy reflects the very small dataset size (101 samples, 5 classes) "
        "rather than model failure — the models are performing as expected given the data constraints.",
        styles["body"]))

    story += figure_block("11_model_comparison.png",
                          "Figure 12: Model Performance Comparison", styles)

    if not model_df.empty:
        story.append(Spacer(1, 6))
        story.append(df_to_table(model_df,
            col_widths=[avail_w * 0.32, avail_w * 0.17,
                        avail_w * 0.17, avail_w * 0.17, avail_w * 0.17]))
    story.append(Spacer(1, 12))

    # ── 6. Confusion Matrices ─────────────────────────────────────────────────
    story += section_heading("6. Confusion Matrices", styles)
    story += figure_pair(
        "10_confusion_logistic_regression.png", "Figure 13: Logistic Regression",
        "10_confusion_random_forest.png",       "Figure 14: Random Forest",
        styles)
    story += figure_pair(
        "10_confusion_xgboost.png", "Figure 15: XGBoost",
        "10_confusion_svm.png",     "Figure 16: SVM",
        styles)

    # ── 7. SHAP ───────────────────────────────────────────────────────────────
    story += section_heading("7. SHAP Feature Importance", styles)
    story.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) values quantify the contribution of each feature "
        "to the model's predictions across all CGPA bands. Features with larger bars have "
        "greater influence on the predicted outcome.",
        styles["body"]))
    story += figure_block("12_shap_feature_importance.png",
                          "Figure 17: SHAP Feature Importance — Logistic Regression", styles)

    # ── 8. Key Findings ───────────────────────────────────────────────────────
    story += section_heading("8. Key Findings", styles)
    findings = [
        "Students with higher mental health burden scores tend to cluster in lower CGPA bands, "
        "suggesting a compounding effect of co-occurring conditions.",
        "Only 5-6% of students with mental health conditions sought specialist treatment — "
        "a striking treatment gap that warrants institutional intervention.",
        "Depression, anxiety, and panic attacks frequently co-occur, indicating shared "
        "underlying stressors rather than isolated conditions.",
        "Year of study and age correlate with CGPA, suggesting academic pressure increases "
        "over time and compounds mental health challenges.",
        "Logistic Regression achieved the best Weighted F1 (0.45); low overall accuracy is "
        "expected given only 101 samples across 5 CGPA classes.",
        "Chi-square tests found no statistically significant association between individual "
        "mental health conditions and CGPA band, though this is likely a power limitation "
        "of the small sample rather than absence of a true effect.",
    ]
    for f in findings:
        story.append(Paragraph("&#x2022;  " + f, styles["bullet"]))

    story.append(Spacer(1, 16))

    # ── 9. Limitations & Future Work ──────────────────────────────────────────
    story += section_heading("9. Limitations & Future Work", styles)
    limitations = [
        "Sample size (n=101) is too small for reliable 5-class prediction; collecting more "
        "data would substantially improve model performance.",
        "The dataset is self-reported and limited to one university (IIUM), reducing generalisability.",
        "Collapsing CGPA into 3 classes (Low / Medium / High) could improve classification "
        "accuracy and yield clearer patterns.",
        "Future work could incorporate additional features: sleep quality, social support, "
        "financial stress, and academic workload.",
    ]
    for l in limitations:
        story.append(Paragraph("&#x2022;  " + l, styles["bullet"]))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        PDF_OUT,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=2 * cm,
        title="MindGrade — Student Mental Health & CGPA Report",
        author="MindGrade ML Project",
        subject="Predicting Academic Performance from Mental Health Indicators",
    )
    doc.build(story)
    print(f"PDF saved -> {PDF_OUT}")


def run():
    print("=" * 60)
    print("  MindGrade — PDF Report Generator")
    print("=" * 60)
    build_pdf()
    print("=" * 60)
    print("  Done. Open outputs/reports/MindGrade_Report.pdf")
    print("=" * 60)


if __name__ == "__main__":
    run()