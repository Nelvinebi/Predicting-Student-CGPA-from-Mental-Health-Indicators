"""
MindGrade — Phase 3: Feature Engineering
=========================================
Input  : data/processed/mindgrade_cleaned.csv
Output : data/processed/mindgrade_features.csv
         data/processed/mindgrade_features_resampled.csv

Steps:
  1. One-hot encode: gender, course
  2. Keep ordinal: year_of_study, cgpa_label
  3. Retain numeric: age, mh_burden_score, binary MH flags
  4. Define feature matrix X and target vector y
  5. Train/test split (80/20, stratified)
  6. Apply SMOTE on training set to handle class imbalance
  7. Export feature sets and split indices
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH    = os.path.join(BASE_DIR, "data", "processed", "mindgrade_cleaned.csv")
FEAT_PATH     = os.path.join(BASE_DIR, "data", "processed", "mindgrade_features.csv")
SMOTE_PATH    = os.path.join(BASE_DIR, "data", "processed", "mindgrade_features_resampled.csv")
SPLIT_PATH    = os.path.join(BASE_DIR, "data", "processed", "train_test_indices.pkl")

RANDOM_STATE  = 42
TEST_SIZE     = 0.20


# ── Step 1: One-hot encode categoricals ───────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = pd.get_dummies(df, columns=["gender", "course"], drop_first=False)
    # Convert bool columns to int
    bool_cols = df_enc.select_dtypes(include="bool").columns
    df_enc[bool_cols] = df_enc[bool_cols].astype(int)
    print(f"[1] After one-hot encoding: {df_enc.shape[1]} columns")
    return df_enc


# ── Step 2: Define X and y ────────────────────────────────────────────────────
def split_features_target(df_enc: pd.DataFrame):
    drop_cols = ["cgpa", "cgpa_label"]   # drop label string; keep cgpa_label as y
    target    = "cgpa_label"

    y = df_enc[target]
    X = df_enc.drop(columns=drop_cols + [target] if target not in drop_cols else drop_cols)

    # Re-include cgpa_label as y (already extracted above)
    # Make sure cgpa (string) is not in X
    if "cgpa" in X.columns:
        X = X.drop(columns=["cgpa"])

    print(f"[2] Feature matrix X: {X.shape}")
    print(f"    Target y — class distribution:\n{y.value_counts().sort_index().to_string()}")
    return X, y


# ── Step 3: Train/test split ──────────────────────────────────────────────────
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[3] Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


# ── Step 4: SMOTE oversampling ────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    print(f"\n[4] Before SMOTE — class distribution:\n{y_train.value_counts().sort_index().to_string()}")

    # k_neighbors must be < smallest class size in training set
    min_class_count = y_train.value_counts().min()
    k = max(1, min_class_count - 1)
    print(f"    Using k_neighbors={k} (smallest class has {min_class_count} sample(s))")

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"    After SMOTE  — class distribution:\n{pd.Series(y_res).value_counts().sort_index().to_string()}")
    print(f"    Resampled training set: {X_res.shape[0]} rows")
    return X_res, y_res


# ── Step 5: Export ────────────────────────────────────────────────────────────
def export_all(X, y, X_train, X_test, y_train, y_test, X_res, y_res, df_enc):
    os.makedirs(os.path.dirname(FEAT_PATH), exist_ok=True)

    # Full encoded feature file (with target)
    full = df_enc.copy()
    full.to_csv(FEAT_PATH, index=False)
    print(f"\n[5] Full encoded dataset → {FEAT_PATH}")

    # Resampled training set
    res_df = pd.DataFrame(X_res, columns=X_train.columns)
    res_df["cgpa_label"] = y_res
    res_df.to_csv(SMOTE_PATH, index=False)
    print(f"    Resampled training set → {SMOTE_PATH}")

    # Save split indices / arrays for Phase 4
    joblib.dump({
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_res": X_res,     "y_res": y_res,
        "feature_names": list(X_train.columns),
    }, SPLIT_PATH)
    print(f"    Train/test split saved → {SPLIT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  MindGrade — Phase 3: Feature Engineering")
    print("=" * 60)

    df     = pd.read_csv(CLEAN_PATH)
    df_enc = encode_features(df)
    X, y   = split_features_target(df_enc)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_res, y_res = apply_smote(X_train, y_train)
    export_all(X, y, X_train, X_test, y_train, y_test, X_res, y_res, df_enc)

    print("\n" + "=" * 60)
    print("  Phase 3 complete.")
    print("=" * 60)
    return X_train, X_test, y_train, y_test, X_res, y_res


if __name__ == "__main__":
    run()