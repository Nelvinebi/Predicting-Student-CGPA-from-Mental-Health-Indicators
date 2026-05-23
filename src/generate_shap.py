"""
Standalone SHAP figure generator for MindGrade.
Run this once to produce 12_shap_feature_importance.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CGPA_LABELS = ["0-1.99", "2.00-2.49", "2.50-2.99", "3.00-3.49", "3.50-4.00"]

# Load data and model
data  = joblib.load(os.path.join(BASE, "data", "processed", "train_test_indices.pkl"))
model = joblib.load(os.path.join(BASE, "outputs", "models", "logistic_regression.pkl"))

feature_names = data["feature_names"]
X_train_df    = pd.DataFrame(data["X_res"],  columns=feature_names)
X_test_df     = pd.DataFrame(data["X_test"], columns=feature_names)

print("Computing SHAP values...")
explainer = shap.LinearExplainer(model, X_train_df)
shap_vals = explainer.shap_values(X_test_df)

# shap_vals shape: (21, 17, 5) — split into list of 5 arrays of shape (21, 17)
shap_list = [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
print(f"SHAP list: {len(shap_list)} classes, each shape {shap_list[0].shape}")

# Compute mean absolute SHAP per feature per class, then sum across classes
mean_abs = np.array([np.abs(s).mean(axis=0) for s in shap_list])  # (5, 17)
total_importance = mean_abs.sum(axis=0)                             # (17,)

# Sort features by total importance
sorted_idx  = np.argsort(total_importance)
sorted_feats = [feature_names[i] for i in sorted_idx]
sorted_vals  = [mean_abs[:, i] for i in sorted_idx]

# Plot stacked horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 8))
colors  = ["#4e89e8", "#f4a261", "#2a9d8f", "#e76f51", "#a8dadc"]
bottoms = np.zeros(len(sorted_feats))

for cls_idx, cls_name in enumerate(CGPA_LABELS):
    vals = np.array([sorted_vals[f][cls_idx] for f in range(len(sorted_feats))])
    ax.barh(sorted_feats, vals, left=bottoms,
            color=colors[cls_idx], label=cls_name, edgecolor="white", linewidth=0.4)
    bottoms += vals

ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=11)
ax.set_title("SHAP Feature Importance — Logistic Regression\n(contribution per CGPA class)",
             fontsize=13, fontweight="bold")
ax.legend(title="CGPA Band", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()

out_path = os.path.join(FIGURES_DIR, "12_shap_feature_importance.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_path}")