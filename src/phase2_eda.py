"""
MindGrade — Phase 2: Exploratory Data Analysis
===============================================
Input  : data/processed/mindgrade_cleaned.csv
Output : outputs/figures/ (PNG charts)

Steps:
  1. Distribution of CGPA bands
  2. Gender breakdown
  3. Mental health prevalence (depression, anxiety, panic attack)
  4. MH burden score distribution
  5. Mental health conditions vs CGPA (stacked bar charts)
  6. Age distribution
  7. Year of study vs CGPA
  8. Course vs CGPA heatmap
  9. Correlation heatmap
  10. Chi-square tests: each MH condition vs CGPA
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import chi2_contingency

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH  = os.path.join(BASE_DIR, "data", "processed", "mindgrade_cleaned.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
CGPA_ORDER = ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"]
MH_COLS    = ["depression", "anxiety", "panic_attack"]
PALETTE    = sns.color_palette("Set2")


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. CGPA distribution ───────────────────────────────────────────────────────
def plot_cgpa_dist(df):
    counts = df["cgpa"].value_counts().reindex(CGPA_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11)
    ax.set_title("CGPA Band Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("CGPA Band")
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, counts.max() + 8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    save(fig, "01_cgpa_distribution.png")


# ── 2. Gender breakdown ────────────────────────────────────────────────────────
def plot_gender(df):
    counts = df["gender"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=[PALETTE[0], PALETTE[1]], startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(12)
    ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "02_gender_distribution.png")


# ── 3. Mental health prevalence ────────────────────────────────────────────────
def plot_mh_prevalence(df):
    labels  = ["Depression", "Anxiety", "Panic Attack"]
    yes_pct = [df[c].mean() * 100 for c in MH_COLS]
    no_pct  = [100 - p for p in yes_pct]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(labels))
    bars_yes = ax.bar(x, yes_pct, label="Yes", color=PALETTE[3], edgecolor="white")
    bars_no  = ax.bar(x, no_pct, bottom=yes_pct, label="No", color=PALETTE[0], edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Mental Health Condition Prevalence", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage of Students")
    ax.legend(title="Reported")
    for bar, pct in zip(bars_yes, yes_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, pct / 2,
                f"{pct:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    save(fig, "03_mh_prevalence.png")


# ── 4. MH burden score ─────────────────────────────────────────────────────────
def plot_burden_score(df):
    counts = df["mh_burden_score"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"Score {i}" for i in counts.index],
        counts.values, color=PALETTE, edgecolor="white"
    )
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_title("Mental Health Burden Score Distribution\n(0 = none, 3 = all three conditions)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, counts.max() + 5)
    plt.tight_layout()
    save(fig, "04_mh_burden_score.png")


# ── 5. MH conditions vs CGPA ──────────────────────────────────────────────────
def plot_mh_vs_cgpa(df):
    labels = {"depression": "Depression", "anxiety": "Anxiety", "panic_attack": "Panic Attack"}
    for col, label in labels.items():
        grp = df.groupby(["cgpa", col]).size().unstack(fill_value=0)
        grp = grp.reindex(CGPA_ORDER)
        grp_pct = grp.div(grp.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Raw counts
        grp.plot(kind="bar", ax=axes[0], color=[PALETTE[0], PALETTE[3]],
                 edgecolor="white", rot=15)
        axes[0].set_title(f"{label} vs CGPA — Counts", fontweight="bold")
        axes[0].set_xlabel("CGPA Band")
        axes[0].set_ylabel("Number of Students")
        axes[0].legend(["No", "Yes"], title=label)

        # Percentage
        grp_pct.plot(kind="bar", ax=axes[1], color=[PALETTE[0], PALETTE[3]],
                     edgecolor="white", rot=15)
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
        axes[1].set_title(f"{label} vs CGPA — Proportions", fontweight="bold")
        axes[1].set_xlabel("CGPA Band")
        axes[1].set_ylabel("Percentage")
        axes[1].legend(["No", "Yes"], title=label)

        plt.suptitle(f"Impact of {label} on CGPA", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        save(fig, f"05_{col}_vs_cgpa.png")


# ── 6. Age distribution ────────────────────────────────────────────────────────
def plot_age(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["age"].value_counts().sort_index().plot(kind="bar", ax=ax,
        color=PALETTE, edgecolor="white")
    ax.set_title("Age Distribution of Respondents", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Students")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    save(fig, "06_age_distribution.png")


# ── 7. Year of study vs CGPA ──────────────────────────────────────────────────
def plot_year_vs_cgpa(df):
    pivot = df.groupby(["year_of_study", "cgpa"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=CGPA_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, edgecolor="white", rot=0)
    ax.set_title("Year of Study vs CGPA Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year of Study")
    ax.set_ylabel("Number of Students")
    ax.legend(title="CGPA Band", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    save(fig, "07_year_vs_cgpa.png")


# ── 8. Course vs CGPA heatmap ─────────────────────────────────────────────────
def plot_course_heatmap(df):
    pivot = df.groupby(["course", "cgpa"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=CGPA_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Course Group vs CGPA Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("CGPA Band")
    ax.set_ylabel("Course Group")
    plt.tight_layout()
    save(fig, "08_course_vs_cgpa_heatmap.png")


# ── 9. Correlation heatmap ─────────────────────────────────────────────────────
def plot_correlation(df):
    num_cols = ["age", "year_of_study", "marital_status",
                "depression", "anxiety", "panic_attack",
                "sought_treatment", "mh_burden_score", "cgpa_label"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, linecolor="white",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "09_correlation_heatmap.png")


# ── 10. Chi-square tests ───────────────────────────────────────────────────────
def chi_square_tests(df):
    print("\n── Chi-Square Tests: Mental Health vs CGPA ──")
    results = []
    for col in MH_COLS + ["mh_burden_score"]:
        ct  = pd.crosstab(df[col], df["cgpa"])
        chi2, p, dof, _ = chi2_contingency(ct)
        sig = "✓ Significant" if p < 0.05 else "✗ Not significant"
        results.append({"Feature": col, "Chi2": round(chi2, 4), "p-value": round(p, 4), "dof": dof, "Result": sig})
        print(f"  {col:<20} χ²={chi2:.4f}  p={p:.4f}  dof={dof}  → {sig}")

    results_df = pd.DataFrame(results)
    out_path   = os.path.join(BASE_DIR, "outputs", "reports", "chi_square_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\n  Chi-square table saved → {out_path}")
    return results_df


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  MindGrade — Phase 2: Exploratory Data Analysis")
    print("=" * 60)

    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded cleaned dataset: {df.shape}\n")

    print("[1] CGPA distribution")
    plot_cgpa_dist(df)

    print("[2] Gender breakdown")
    plot_gender(df)

    print("[3] MH prevalence")
    plot_mh_prevalence(df)

    print("[4] MH burden score")
    plot_burden_score(df)

    print("[5] MH conditions vs CGPA")
    plot_mh_vs_cgpa(df)

    print("[6] Age distribution")
    plot_age(df)

    print("[7] Year of study vs CGPA")
    plot_year_vs_cgpa(df)

    print("[8] Course vs CGPA heatmap")
    plot_course_heatmap(df)

    print("[9] Correlation heatmap")
    plot_correlation(df)

    print("[10] Chi-square tests")
    chi_square_tests(df)

    print("\n" + "=" * 60)
    print("  Phase 2 complete. Figures saved to outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    run()