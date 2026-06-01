"""
================================================================================
MindGrade — Docstrings for all source modules
================================================================================
Copy the relevant docstring block into each corresponding file in src/.
Each block includes:
  - A module-level docstring  →  paste at the very top of the file
  - Function-level docstrings →  paste inside each function definition
================================================================================
"""


# ══════════════════════════════════════════════════════════════════════════════
# FILE: src/phase1_data_cleaning.py
# ══════════════════════════════════════════════════════════════════════════════

MODULE_DOCSTRING_PHASE1 = '''
"""
phase1_data_cleaning.py — MindGrade Pipeline · Phase 1
=======================================================
Ingests the raw IIUM Student Mental Health survey CSV and produces a fully
cleaned, standardised dataset ready for exploratory analysis.

Input
-----
data/raw/Student_Mental_health.csv
    Raw Google Forms export (101 rows × 11 columns). Contains inconsistent
    casing, 49 raw course-name variants, whitespace-padded CGPA bands,
    Yes/No string responses, and 1 missing age value.

Output
------
data/processed/mindgrade_cleaned.csv
    Cleaned dataset (101 rows × 12 columns) with:
    - Snake_case column names
    - 49 course variants collapsed into 7 faculty categories
    - Year-of-study normalised to integer (1–4)
    - CGPA bands stripped of whitespace + ordinal cgpa_label added (0–4)
    - Yes/No columns converted to binary int flags (1/0)
    - 1 missing age imputed with column median (19.0)
    - Composite mh_burden_score engineered (depression + anxiety + panic_attack)

Usage
-----
    python src/phase1_data_cleaning.py

Dependencies
------------
    pandas, numpy
"""
'''

FUNCTION_DOCSTRINGS_PHASE1 = '''
def rename_columns(df):
    """
    Rename raw survey column headers to clean snake_case identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with original Google Forms column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardised column names.
    """

def group_courses(course_str):
    """
    Map a raw course name string to one of 7 standardised faculty categories.

    Handles 49 known variants including inconsistent casing (e.g. "koe",
    "KOE", "Koe") and abbreviated entries.

    Parameters
    ----------
    course_str : str
        Raw course name from the survey response.

    Returns
    -------
    str
        One of: "Computing & IT", "Engineering", "Islamic Studies",
        "Social Sciences", "Life Sciences", "Business & Economics",
        "Sciences", or "Other" if unrecognised.
    """

def encode_binary_columns(df, columns):
    """
    Convert Yes/No string responses to binary integer flags (1/0).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the target columns.
    columns : list of str
        Column names to encode (e.g. ["depression", "anxiety",
        "panic_attack", "sought_treatment", "marital_status"]).

    Returns
    -------
    pd.DataFrame
        Dataframe with the specified columns replaced by int64 0/1 values.
    """

def engineer_burden_score(df):
    """
    Create the composite Mental Health Burden Score feature.

    Sums the three binary mental health flags (depression, anxiety,
    panic_attack) into a single ordinal score ranging from 0 (no
    conditions) to 3 (all three conditions present).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing encoded depression, anxiety, and
        panic_attack columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with a new "mh_burden_score" column (int, 0–3).
    """

def run_cleaning_pipeline(input_path, output_path):
    """
    Execute the full Phase 1 cleaning pipeline end-to-end.

    Reads the raw CSV, applies all cleaning and engineering steps in order,
    and writes the processed dataset to disk.

    Parameters
    ----------
    input_path : str or Path
        Path to the raw survey CSV file.
    output_path : str or Path
        Destination path for the cleaned CSV output.

    Returns
    -------
    pd.DataFrame
        The cleaned and feature-engineered dataframe (also saved to disk).

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    """
'''


# ══════════════════════════════════════════════════════════════════════════════
# FILE: src/phase2_eda.py
# ══════════════════════════════════════════════════════════════════════════════

MODULE_DOCSTRING_PHASE2 = '''
"""
phase2_eda.py — MindGrade Pipeline · Phase 2
=============================================
Performs exploratory data analysis on the cleaned dataset and runs
chi-square independence tests between mental health features and CGPA band.

Input
-----
data/processed/mindgrade_cleaned.csv
    Output of Phase 1. Expected shape: 101 rows × 12 columns.

Outputs
-------
outputs/figures/01_cgpa_distribution.png
    Bar chart of CGPA band counts.
outputs/figures/02_gender_distribution.png
    Pie chart of gender breakdown.
outputs/figures/03_mh_prevalence.png
    Stacked bar chart of mental health condition prevalence vs sought treatment.
outputs/figures/04_mh_burden_score.png
    Distribution of the composite MH burden score (0–3).
outputs/figures/05_depression_vs_cgpa.png
    Side-by-side count and proportion charts for depression across CGPA bands.
outputs/figures/05_anxiety_vs_cgpa.png
    Side-by-side count and proportion charts for anxiety across CGPA bands.
outputs/figures/05_panic_attack_vs_cgpa.png
    Side-by-side count and proportion charts for panic attacks across CGPA bands.
outputs/figures/06_age_distribution.png
    Bar chart of student age distribution.
outputs/figures/07_year_vs_cgpa.png
    Grouped bar chart of year-of-study vs CGPA band.
outputs/figures/08_course_vs_cgpa_heatmap.png
    Heatmap of student counts per course group × CGPA band.
outputs/figures/09_correlation_heatmap.png
    Lower-triangular Pearson correlation heatmap across all numeric features.
outputs/reports/chi_square_results.csv
    Table of chi-square statistics, p-values, and degrees of freedom for each
    mental health feature vs CGPA band.

Usage
-----
    python src/phase2_eda.py

Dependencies
------------
    pandas, matplotlib, seaborn, scipy
"""
'''

FUNCTION_DOCSTRINGS_PHASE2 = '''
def plot_cgpa_distribution(df, output_dir):
    """
    Plot and save a bar chart of the CGPA band distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing a "cgpa" column with band labels.
    output_dir : str or Path
        Directory where the figure PNG will be saved.

    Returns
    -------
    None
    """

def plot_mh_vs_cgpa(df, condition, output_dir):
    """
    Plot side-by-side count and proportion charts for one mental health
    condition across CGPA bands.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with the given condition column and "cgpa" column.
    condition : str
        Column name of the binary mental health flag to plot
        (e.g. "depression", "anxiety", "panic_attack").
    output_dir : str or Path
        Directory where the figure PNG will be saved.

    Returns
    -------
    None
    """

def run_chi_square_tests(df, features, target, output_path):
    """
    Run chi-square tests of independence between each feature and the target.

    Uses scipy.stats.chi2_contingency on a crosstab of each feature vs the
    target column. Results are printed to stdout and saved to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing all feature and target columns.
    features : list of str
        Column names to test against the target
        (e.g. ["depression", "anxiety", "panic_attack", "mh_burden_score"]).
    target : str
        Target column name (e.g. "cgpa").
    output_path : str or Path
        Destination path for the chi_square_results.csv output.

    Returns
    -------
    pd.DataFrame
        Results table with columns: feature, chi2, p_value, dof, significant.
    """
'''


# ══════════════════════════════════════════════════════════════════════════════
# FILE: src/phase3_feature_engineering.py
# ══════════════════════════════════════════════════════════════════════════════

MODULE_DOCSTRING_PHASE3 = '''
"""
phase3_feature_engineering.py — MindGrade Pipeline · Phase 3
=============================================================
One-hot encodes categorical features, performs an 80/20 stratified
train/test split, and applies adaptive SMOTE to the training set to
address severe CGPA class imbalance.

Input
-----
data/processed/mindgrade_cleaned.csv
    Output of Phase 1. Expected shape: 101 rows × 12 columns.

Outputs
-------
data/processed/mindgrade_features.csv
    Full one-hot encoded feature matrix (101 rows × ~17 columns).
data/processed/mindgrade_features_resampled.csv
    SMOTE-resampled training set (190 rows × ~17 columns).
data/processed/train_test_indices.pkl
    Serialised dict containing:
        X_train, X_test  →  np.ndarray of feature values
        y_train, y_test  →  np.ndarray of CGPA labels
        feature_names    →  list of column names after encoding

Usage
-----
    python src/phase3_feature_engineering.py

Notes
-----
SMOTE is applied with adaptive k_neighbors = min_class_size − 1 to avoid
crashes when the smallest training class has fewer than 5 samples (the
SMOTE default). This expands the 80-sample training set to 190 samples
(5 classes × 38 synthetic samples each).

IMPORTANT: SMOTE is applied AFTER splitting. Applying it before the split
would cause data leakage and inflate all downstream evaluation metrics.

Dependencies
------------
    pandas, numpy, scikit-learn, imbalanced-learn, joblib
"""
'''

FUNCTION_DOCSTRINGS_PHASE3 = '''
def encode_features(df):
    """
    Apply one-hot encoding to categorical columns and convert bool columns
    to int, returning the full feature matrix and target series.

    One-hot encodes: "gender", "course".
    Drops: the original "cgpa" string column and "cgpa_label" (kept as target).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from Phase 1.

    Returns
    -------
    X : pd.DataFrame
        Encoded feature matrix (~17 columns).
    y : pd.Series
        Target series of ordinal CGPA labels (int, 0–4).
    """

def apply_smote(X_train, y_train):
    """
    Apply adaptive SMOTE oversampling to the training set.

    Dynamically computes k_neighbors = min_class_size − 1 to handle
    classes with as few as 2 samples without raising a ValueError.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (pre-SMOTE).
    y_train : np.ndarray
        Training labels (pre-SMOTE).

    Returns
    -------
    X_resampled : np.ndarray
        Oversampled training feature matrix.
    y_resampled : np.ndarray
        Oversampled training labels (balanced across all 5 CGPA classes).

    Notes
    -----
    The resampled training set will contain n_classes × max_class_size rows.
    Original class proportions are NOT preserved — all classes are upsampled
    to match the majority class.
    """
'''


# ══════════════════════════════════════════════════════════════════════════════
# FILE: src/phase4_modelling.py
# ══════════════════════════════════════════════════════════════════════════════

MODULE_DOCSTRING_PHASE4 = '''
"""
phase4_modelling.py — MindGrade Pipeline · Phase 4
===================================================
Trains four classifiers on the SMOTE-resampled training data, evaluates
each on the held-out test set, generates confusion matrices, and exports
a model performance comparison chart and CSV.

Input
-----
data/processed/train_test_indices.pkl
    Serialised train/test arrays and feature names from Phase 3.

Outputs
-------
outputs/models/logistic_regression.pkl
outputs/models/random_forest.pkl
outputs/models/xgboost.pkl
outputs/models/svm.pkl
    Trained, serialised model objects (joblib format).
outputs/figures/10_confusion_logistic_regression.png
outputs/figures/10_confusion_random_forest.png
outputs/figures/10_confusion_xgboost.png
outputs/figures/10_confusion_svm.png
    Per-model confusion matrices scoped to classes present in predictions.
outputs/figures/11_model_comparison.png
    Side-by-side bar chart of Accuracy, Macro F1, and Weighted F1 per model.
outputs/reports/model_results.csv
    Table of Accuracy, Macro F1, Weighted F1, and ROC-AUC per model.

Usage
-----
    python src/phase4_modelling.py

Notes
-----
ROC-AUC may be undefined (NaN) if any CGPA class has 0 test samples —
this is a dataset limitation (class 1: 2.00–2.49 has only 2 total samples),
not a code error. The evaluation catches this gracefully.

All models are trained with random_state=42 for reproducibility.

Dependencies
------------
    pandas, numpy, scikit-learn, xgboost, joblib, matplotlib, seaborn
"""
'''

FUNCTION_DOCSTRINGS_PHASE4 = '''
def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test,
                       feature_names, output_dir):
    """
    Fit a classifier, evaluate it on the test set, and save its confusion matrix.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Unfitted classifier instance (e.g. LogisticRegression()).
    model_name : str
        Human-readable name used for filenames and report labels
        (e.g. "logistic_regression").
    X_train : np.ndarray
        SMOTE-resampled training features.
    y_train : np.ndarray
        SMOTE-resampled training labels.
    X_test : np.ndarray
        Held-out test features (21 samples).
    y_test : np.ndarray
        Held-out test labels.
    feature_names : list of str
        Column names of X_train/X_test, used for SHAP compatibility.
    output_dir : str or Path
        Directory where the confusion matrix PNG and .pkl model will be saved.

    Returns
    -------
    dict
        Dictionary with keys: model_name, accuracy, macro_f1, weighted_f1,
        roc_auc (float or NaN if undefined).
    """

def plot_model_comparison(results_df, output_path):
    """
    Plot a grouped bar chart comparing all four models across key metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe with columns: model_name, accuracy, macro_f1, weighted_f1.
        Typically the output of collecting results from train_and_evaluate().
    output_path : str or Path
        Destination path for the comparison chart PNG.

    Returns
    -------
    None
    """
'''


# ══════════════════════════════════════════════════════════════════════════════
# FILE: src/generate_shap.py
# ══════════════════════════════════════════════════════════════════════════════

MODULE_DOCSTRING_SHAP = '''
"""
generate_shap.py — MindGrade · SHAP Feature Importance
=======================================================
Loads the best-performing trained model (Logistic Regression), computes
SHAP values using LinearExplainer, and renders a stacked horizontal bar
chart showing mean absolute SHAP contribution per feature per CGPA class.

Input
-----
outputs/models/logistic_regression.pkl
    Trained LogisticRegression model serialised by Phase 4.
data/processed/train_test_indices.pkl
    Serialised train/test arrays and feature names from Phase 3.
    X_train is used as the SHAP background dataset.

Output
------
outputs/figures/12_shap_feature_importance.png
    Stacked horizontal bar chart. Each bar segment represents the mean
    absolute SHAP value contributed by one CGPA class (0–4) for a given
    feature. Features are sorted by total importance descending.

Usage
-----
    python src/generate_shap.py

Notes
-----
The raw SHAP output has shape (n_test_samples, n_features, n_classes),
i.e. (21, 17, 5). This is decomposed by taking the mean absolute value
across samples for each (feature, class) pair, then stacked per class.

Dependencies
------------
    shap, joblib, numpy, pandas, matplotlib
"""
'''

FUNCTION_DOCSTRINGS_SHAP = '''
def compute_shap_values(model, X_background, X_explain):
    """
    Compute SHAP values for a multi-class linear model.

    Uses shap.LinearExplainer with the training set as the background
    distribution. Returns raw SHAP values of shape
    (n_samples, n_features, n_classes).

    Parameters
    ----------
    model : sklearn LogisticRegression
        Fitted multi-class logistic regression model.
    X_background : np.ndarray
        Background dataset used to compute expected SHAP values
        (typically X_train, shape: n_train × n_features).
    X_explain : np.ndarray
        Samples to explain (typically X_test, shape: n_test × n_features).

    Returns
    -------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features, n_classes).
    """

def plot_shap_bar(shap_values, feature_names, class_labels, output_path):
    """
    Render and save a stacked horizontal bar chart of mean absolute SHAP values.

    Each bar represents a feature; each segment within a bar represents the
    mean absolute SHAP contribution from one CGPA class.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP array of shape (n_samples, n_features, n_classes).
    feature_names : list of str
        Names of the features (x-axis labels), length must equal n_features.
    class_labels : list of str
        Display labels for each CGPA class, e.g.
        ["0–1.99", "2.00–2.49", "2.50–2.99", "3.00–3.49", "3.50–4.00"].
    output_path : str or Path
        Destination path for the chart PNG.

    Returns
    -------
    None
    """
'''


# ══════════════════════════════════════════════════════════════════════════════
# FILE: src/phase5_reporting.py
# ══════════════════════════════════════════════════════════════════════════════

MODULE_DOCSTRING_PHASE5 = '''
"""
phase5_reporting.py — MindGrade Pipeline · Phase 5
===================================================
Generates a self-contained HTML research report by embedding all output
figures and CSV result tables as base64-encoded data URIs. The report
requires no external assets and can be opened in any browser.

Inputs
------
outputs/figures/*.png
    All 17 figure PNGs generated by Phases 2, 4, and generate_shap.py.
outputs/reports/chi_square_results.csv
    Chi-square test results table from Phase 2.
outputs/reports/model_results.csv
    Model performance comparison table from Phase 4.

Output
------
outputs/reports/MindGrade_Report.html
    Single-file HTML report with embedded figures, styled tables,
    and key findings narrative. No external dependencies.

Usage
-----
    python src/phase5_reporting.py

Dependencies
------------
    pandas, base64, pathlib (stdlib only — no third-party HTML libraries)
"""
'''

FUNCTION_DOCSTRINGS_PHASE5 = '''
def embed_image_as_base64(image_path):
    """
    Read a PNG file and return an HTML <img> tag with a base64 data URI src.

    This embeds the image directly into the HTML so the report is fully
    self-contained with no external file dependencies.

    Parameters
    ----------
    image_path : str or Path
        Path to the PNG image file.

    Returns
    -------
    str
        HTML string: <img src="data:image/png;base64,..." .../>

    Raises
    ------
    FileNotFoundError
        If image_path does not exist.
    """

def build_html_report(figures_dir, reports_dir, output_path):
    """
    Assemble and write the complete HTML report to disk.

    Iterates through all expected figure filenames in order, embeds each
    as a base64 image, and inserts the chi-square and model results tables
    as styled HTML tables generated from the CSV files.

    Parameters
    ----------
    figures_dir : str or Path
        Directory containing output figure PNGs (outputs/figures/).
    reports_dir : str or Path
        Directory containing chi_square_results.csv and model_results.csv.
    output_path : str or Path
        Destination path for the HTML report file.

    Returns
    -------
    None
    """
'''


# ══════════════════════════════════════════════════════════════════════════════
# FILE: src/generate_pdf.py
# ══════════════════════════════════════════════════════════════════════════════

MODULE_DOCSTRING_PDF = '''
"""
generate_pdf.py — MindGrade · PDF Report Generator
===================================================
Programmatically assembles a paginated PDF research report using ReportLab.
The report includes a styled cover page, stat cards, all 17 output figures
arranged in pairs, formatted results tables, key findings narrative, and
page-numbered footers — with no browser or HTML rendering dependency.

Inputs
------
outputs/figures/*.png
    All 17 figure PNGs generated by Phases 2, 4, and generate_shap.py.
outputs/reports/chi_square_results.csv
    Chi-square test results from Phase 2.
outputs/reports/model_results.csv
    Model performance comparison from Phase 4.

Output
------
outputs/reports/MindGrade_Report.pdf
    Full paginated PDF research report (~20–25 pages).

Usage
-----
    python src/generate_pdf.py

Notes
-----
All layout dimensions are in ReportLab points (1 pt = 1/72 inch).
The document uses A4 page size. Figures are scaled to fit within
a two-column layout while preserving aspect ratio.

Dependencies
------------
    reportlab, pandas, pathlib
"""
'''

FUNCTION_DOCSTRINGS_PDF = '''
def build_cover_page(canvas, doc, title, subtitle, author):
    """
    Render the report cover page onto the given ReportLab canvas.

    Draws the project title, subtitle, author name, date, and a
    decorative header band using ReportLab primitives.

    Parameters
    ----------
    canvas : reportlab.pdfgen.canvas.Canvas
        Active canvas object for the current page.
    doc : reportlab.platypus.BaseDocTemplate
        The parent document template (used for page dimensions).
    title : str
        Main report title displayed prominently on the cover.
    subtitle : str
        Secondary descriptor line below the title.
    author : str
        Author name displayed at the bottom of the cover.

    Returns
    -------
    None
    """

def add_figure_pair(story, left_path, right_path, left_caption, right_caption):
    """
    Append a two-column figure pair to the ReportLab story list.

    Each figure is scaled proportionally to fit half the page width.
    Captions are rendered in italic below each figure.

    Parameters
    ----------
    story : list
        ReportLab flowable list being built for the document.
    left_path : str or Path
        Path to the left-column figure PNG.
    right_path : str or Path
        Path to the right-column figure PNG. Pass None to render a
        single centred figure spanning both columns.
    left_caption : str
        Caption text for the left figure.
    right_caption : str
        Caption text for the right figure (ignored if right_path is None).

    Returns
    -------
    None
        Modifies story in-place.
    """

def df_to_reportlab_table(df, col_widths=None):
    """
    Convert a pandas DataFrame to a styled ReportLab Table flowable.

    The header row is rendered with a dark background and white bold text.
    Body rows alternate between white and light grey for readability.

    Parameters
    ----------
    df : pd.DataFrame
        Data to render. Column names become the header row.
    col_widths : list of float, optional
        Width of each column in ReportLab points. If None, columns are
        distributed evenly across the page width.

    Returns
    -------
    reportlab.platypus.Table
        Styled Table flowable ready to append to a story list.
    """
'''


# ══════════════════════════════════════════════════════════════════════════════
# USAGE INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

INSTRUCTIONS = """
HOW TO USE THIS FILE
====================

For each source file, do the following:

1. Open the corresponding phase file in src/.

2. Paste the MODULE_DOCSTRING at the very top of the file, immediately
   after any existing `import` statements — or ideally BEFORE them
   (module docstrings conventionally go first, before imports).

   Example structure:
   ------------------
   \"\"\"
   phase1_data_cleaning.py — MindGrade Pipeline · Phase 1
   ...
   \"\"\"

   import pandas as pd
   import numpy as np
   ...

3. For each function listed in FUNCTION_DOCSTRINGS, paste the docstring
   on the first line INSIDE the function body, before any code:

   Example:
   --------
   def rename_columns(df):
       \"\"\"
       Rename raw survey column headers to clean snake_case identifiers.
       ...
       \"\"\"
       # your existing code here
       df.columns = [...]
       return df

4. If a function in your actual code has a different name or signature
   than what's listed here, adjust the Parameters/Returns sections to match.

DOCSTRING STYLE: Google-style with Parameters / Returns / Notes sections.
This is compatible with Sphinx autodoc and mkdocstrings if you ever want
to generate HTML documentation automatically from your code.
"""

print(INSTRUCTIONS)
