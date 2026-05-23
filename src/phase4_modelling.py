"""
MindGrade — Phase 4: Modelling & Evaluation
============================================
Input  : data/processed/train_test_indices.pkl  (from Phase 3)
Output : outputs/models/  (saved .pkl model files)
         outputs/figures/ (confusion matrices, ROC curves, SHAP plots)
         outputs/reports/model_results.csv

Models trained:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost
  4. Support Vector Machine (SVM)

Evaluation metrics:
  - Accuracy, Macro F1, Weighted F1
  - Confusion matrix (per model)
  - ROC-AUC (one-vs-rest, per model)
  - SHAP feature importance (best model only)
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.metrics       import (accuracy_score, f1_score,
                                   confusion_matrix,
                                   classification_report,
                                   roc_auc_score)
from sklearn.preprocessing import label_binarize
from xgboost               import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH  = os.path.join(BASE_DIR, "data", "processed", "train_test_indices.pkl")
MODELS_DIR  = os.path.join(BASE_DIR, "outputs", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
ALL_CLASSES  = [0, 1, 2, 3, 4]
N_CLASSES    = 5
CGPA_LABELS  = {0: "0–1.99", 1: "2.00–2.49", 2: "2.50–2.99", 3: "3.00–3.49", 4: "3.50–4.00"}


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {path}")

def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {path}")


# ── Load split data ────────────────────────────────────────────────────────────
def load_data():
    data = joblib.load(SPLIT_PATH)
    print(f"[Load] Features: {data['X_res'].shape[1]} | "
          f"Train (resampled): {data['X_res'].shape[0]} | "
          f"Test: {data['X_test'].shape[0]}")
    return data


# ── Define models ──────────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, multi_class="multinomial"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE, verbosity=0
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ),
    }


# ── Train & evaluate ───────────────────────────────────────────────────────────
def train_evaluate(models, data):
    X_train = data["X_res"]
    y_train = data["y_res"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    results = []

    for name, model in models.items():
        print(f"\n── Training: {name} ──")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro",    labels=ALL_CLASSES, zero_division=0)
        f1_wt  = f1_score(y_test, y_pred, average="weighted", labels=ALL_CLASSES, zero_division=0)

        # ROC-AUC — pad probability matrix to cover all 5 classes
        try:
            # model.classes_ may be a subset; build full 5-col prob matrix
            prob_full = np.zeros((len(y_test), N_CLASSES))
            for i, cls in enumerate(model.classes_):
                prob_full[:, cls] = y_prob[:, i]

            y_bin   = label_binarize(y_test, classes=ALL_CLASSES)
            roc_auc = roc_auc_score(y_bin, prob_full, multi_class="ovr",
                                    average="macro", labels=ALL_CLASSES)
        except Exception as e:
            roc_auc = float("nan")
            print(f"  ROC-AUC skipped: {e}")

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {f1_mac:.4f}")
        print(f"  Weighted F1: {f1_wt:.4f}")
        print(f"  ROC-AUC:     {roc_auc:.4f}")

        # Build target_names only for classes present in y_test or y_pred
        present = sorted(set(y_test) | set(y_pred))
        tgt_names = [CGPA_LABELS[c] for c in present]
        print(classification_report(y_test, y_pred,
              labels=present, target_names=tgt_names, zero_division=0))

        results.append({
            "Model": name, "Accuracy": round(acc, 4),
            "Macro F1": round(f1_mac, 4), "Weighted F1": round(f1_wt, 4),
            "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else "n/a",
        })

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"  Model saved → {model_path}")

        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred, name)

    return results, models


# ── Confusion matrix plot ──────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name):
    present   = sorted(set(y_test) | set(y_pred))
    tgt_names = [CGPA_LABELS[c] for c in present]
    cm        = confusion_matrix(y_test, y_pred, labels=present)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=tgt_names, yticklabels=tgt_names,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted CGPA Band")
    ax.set_ylabel("Actual CGPA Band")
    plt.xticks(rotation=25)
    plt.tight_layout()
    fname = f"10_confusion_{model_name.replace(' ', '_').lower()}.png"
    save_fig(fig, fname)


# ── Model comparison bar chart ─────────────────────────────────────────────────
def plot_model_comparison(results):
    df_res = pd.DataFrame(results).set_index("Model")
    metrics = ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(10, 6))
    df_res[metrics].plot(kind="bar", ax=ax, edgecolor="white", rot=15)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", label="0.5 baseline")
    plt.tight_layout()
    save_fig(fig, "11_model_comparison.png")


# ── SHAP feature importance ────────────────────────────────────────────────────
def plot_shap(best_model_name, models, data):
    print(f"\n── SHAP Explanation for: {best_model_name} ──")
    model         = models[best_model_name]
    feature_names = data["feature_names"]
    X_train_df    = pd.DataFrame(data["X_res"],  columns=feature_names)
    X_test_df     = pd.DataFrame(data["X_test"], columns=feature_names)

    try:
        if best_model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test_df)
            # shap_vals is already a list of 2D arrays [n_samples, n_features] per class
            shap_list = shap_vals

        elif best_model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_train_df)
            shap_vals = explainer.shap_values(X_test_df)
            # shap_vals shape: (n_samples, n_features, n_classes) → split into list
            if hasattr(shap_vals, "shape") and len(shap_vals.shape) == 3:
                shap_list = [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
            else:
                shap_list = shap_vals

        else:
            X_background = shap.sample(X_train_df, 50)
            explainer    = shap.KernelExplainer(model.predict_proba, X_background)
            shap_vals    = explainer.shap_values(X_test_df)
            shap_list    = shap_vals

        class_names = [CGPA_LABELS[c] for c in ALL_CLASSES]

        shap.summary_plot(
            shap_list,
            X_test_df,
            plot_type="bar",
            class_names=class_names,
            show=False
        )
        plt.title(f"SHAP Feature Importance — {best_model_name}", fontweight="bold")
        plt.tight_layout()
        save_fig(plt.gcf(), "12_shap_feature_importance.png")
        print("  SHAP plot saved.")

    except Exception as e:
        print(f"  SHAP plot failed: {e}")


# ── Save results CSV ───────────────────────────────────────────────────────────
def save_results(results):
    df_res   = pd.DataFrame(results)
    out_path = os.path.join(REPORTS_DIR, "model_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\n[Results] Saved → {out_path}")
    print(df_res.to_string(index=False))
    best = df_res.loc[df_res["Weighted F1"].idxmax(), "Model"]
    print(f"\n  Best model (Weighted F1): {best}")
    return best


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  MindGrade — Phase 4: Modelling & Evaluation")
    print("=" * 60)

    data    = load_data()
    models  = get_models()
    results, trained_models = train_evaluate(models, data)
    plot_model_comparison(results)
    best_model_name = save_results(results)

    # SHAP only for tree-based best model
    if best_model_name in ["Random Forest", "XGBoost"]:
        plot_shap(best_model_name, trained_models, data)

    print("\n" + "=" * 60)
    print("  Phase 4 complete.")
    print("=" * 60)


if __name__ == "__main__":
    run()