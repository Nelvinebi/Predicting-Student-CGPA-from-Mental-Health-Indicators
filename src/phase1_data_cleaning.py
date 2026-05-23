"""
MindGrade — Phase 1: Data Cleaning & Preprocessing
====================================================
Input  : data/raw/Student_Mental_health.csv
Output : data/processed/mindgrade_cleaned.csv

Steps:
  1. Rename columns to clean snake_case names
  2. Normalize and group course names (49 → 7 categories)
  3. Normalize year_of_study (casing fix → integer 1–4)
  4. Standardize CGPA bands + add ordinal label (cgpa_label)
  5. Binary-encode Yes/No columns
  6. Impute 1 missing age value with median
  7. Engineer mental health burden score (mh_burden_score)
  8. Drop the Timestamp column
  9. Export cleaned dataset
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH   = os.path.join(BASE_DIR, "data", "raw",       "Student_Mental_health.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "mindgrade_cleaned.csv")


# ── Step 1: Load ───────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[1] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── Step 2: Rename columns ─────────────────────────────────────────────────────
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        "timestamp", "gender", "age", "course",
        "year_of_study", "cgpa", "marital_status",
        "depression", "anxiety", "panic_attack", "sought_treatment",
    ]
    print("[2] Columns renamed to snake_case")
    return df


# ── Step 3: Normalize gender ───────────────────────────────────────────────────
def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    df["gender"] = df["gender"].str.strip().str.title()
    print(f"[3] Gender values: {df['gender'].value_counts().to_dict()}")
    return df


# ── Step 4: Normalize year of study ───────────────────────────────────────────
def clean_year(df: pd.DataFrame) -> pd.DataFrame:
    df["year_of_study"] = df["year_of_study"].str.strip().str.lower()
    year_map = {"year 1": 1, "year 2": 2, "year 3": 3, "year 4": 4}
    df["year_of_study"] = df["year_of_study"].map(year_map)
    print(f"[4] Year of study normalized: {df['year_of_study'].value_counts().to_dict()}")
    return df


# ── Step 5: Group & clean course names ────────────────────────────────────────
COURSE_MAP = {
    # Engineering
    "Engineering": "Engineering", "Engine": "Engineering",
    "engin": "Engineering", "ENM": "Engineering",
    # Computing & IT
    "BCS": "Computing & IT", "BIT": "Computing & IT",
    "IT": "Computing & IT", "CTS": "Computing & IT",
    # Islamic Studies
    "Islamic education": "Islamic Studies", "Islamic Education": "Islamic Studies",
    "Pendidikan islam": "Islamic Studies", "Pendidikan Islam": "Islamic Studies",
    "Usuluddin": "Islamic Studies", "Fiqh fatwa": "Islamic Studies",
    "Fiqh": "Islamic Studies", "IRKHS": "Islamic Studies",
    "Irkhs": "Islamic Studies", "KIRKHS": "Islamic Studies",
    "Kirkhs": "Islamic Studies", "Koe": "Islamic Studies",
    "KOE": "Islamic Studies", "koe": "Islamic Studies",
    "KENMS": "Islamic Studies", "MHSC": "Islamic Studies",
    "Malcom": "Islamic Studies", "Kop": "Islamic Studies",
    # Life Sciences
    "Biomedical science": "Life Sciences", "Biotechnology": "Life Sciences",
    "Marine science": "Life Sciences", "Nursing": "Life Sciences",
    "Diploma Nursing": "Life Sciences", "Radiography": "Life Sciences",
    # Social Sciences & Humanities
    "Psychology": "Social Sciences", "psychology": "Social Sciences",
    "Human Resources": "Social Sciences", "Human Sciences": "Social Sciences",
    "Communication": "Social Sciences", "Laws": "Social Sciences",
    "Law": "Social Sciences", "ALA": "Social Sciences",
    "TAASL": "Social Sciences", "BENL": "Social Sciences",
    "Benl": "Social Sciences", "DIPLOMA TESL": "Social Sciences",
    # Business & Economics
    "Accounting": "Business & Economics", "Banking Studies": "Business & Economics",
    "Business Administration": "Business & Economics", "Econs": "Business & Economics",
    # Sciences
    "Mathemathics": "Sciences",
}

def clean_course(df: pd.DataFrame) -> pd.DataFrame:
    df["course"] = df["course"].str.strip().map(COURSE_MAP).fillna("Other")
    print(f"[5] Courses grouped: {df['course'].value_counts().to_dict()}")
    return df


# ── Step 6: Standardize CGPA bands ────────────────────────────────────────────
CGPA_LABEL_MAP = {
    "0 - 1.99":    "0 - 1.99",
    "2.00 - 2.49": "2.00 - 2.49",
    "2.50 - 2.99": "2.50 - 2.99",
    "3.00 - 3.49": "3.00 - 3.49",
    "3.50 - 4.00": "3.50 - 4.00",
}
CGPA_ORDINAL = {
    "0 - 1.99": 0, "2.00 - 2.49": 1,
    "2.50 - 2.99": 2, "3.00 - 3.49": 3, "3.50 - 4.00": 4,
}

def clean_cgpa(df: pd.DataFrame) -> pd.DataFrame:
    df["cgpa"]       = df["cgpa"].str.strip().map(CGPA_LABEL_MAP)
    df["cgpa_label"] = df["cgpa"].map(CGPA_ORDINAL)
    print(f"[6] CGPA distribution: {df['cgpa'].value_counts().to_dict()}")
    return df


# ── Step 7: Binary-encode Yes/No columns ──────────────────────────────────────
BINARY_COLS = ["marital_status", "depression", "anxiety", "panic_attack", "sought_treatment"]

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"[7] Binary encoding applied to: {BINARY_COLS}")
    return df


# ── Step 8: Impute missing age ─────────────────────────────────────────────────
def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    missing = df["age"].isna().sum()
    if missing > 0:
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age)
        print(f"[8] Imputed {missing} missing age value(s) with median ({median_age})")
    else:
        print("[8] No missing age values")
    return df


# ── Step 9: Engineer mental health burden score ────────────────────────────────
def add_mh_burden(df: pd.DataFrame) -> pd.DataFrame:
    df["mh_burden_score"] = df["depression"] + df["anxiety"] + df["panic_attack"]
    print(f"[9] MH burden score distribution: {df['mh_burden_score'].value_counts().sort_index().to_dict()}")
    return df


# ── Step 10: Drop timestamp ────────────────────────────────────────────────────
def drop_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["timestamp"])
    print("[10] Timestamp column dropped")
    return df


# ── Step 11: Validate & export ─────────────────────────────────────────────────
def export(df: pd.DataFrame, path: str) -> None:
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"[!] WARNING: {missing} missing values remain in cleaned dataset")
    else:
        print(f"[11] Validation passed — 0 missing values")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[12] Cleaned dataset saved → {path}")
    print(f"     Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"     Columns: {df.columns.tolist()}")


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  MindGrade — Phase 1: Data Cleaning")
    print("=" * 60)

    df = load_data(RAW_PATH)
    df = rename_columns(df)
    df = clean_gender(df)
    df = clean_year(df)
    df = clean_course(df)
    df = clean_cgpa(df)
    df = encode_binary(df)
    df = impute_age(df)
    df = add_mh_burden(df)
    df = drop_timestamp(df)
    export(df, CLEAN_PATH)

    print("=" * 60)
    print("  Phase 1 complete.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run()
