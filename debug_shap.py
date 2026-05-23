import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data  = joblib.load(os.path.join(BASE, 'data', 'processed', 'train_test_indices.pkl'))
model = joblib.load(os.path.join(BASE, 'outputs', 'models', 'logistic_regression.pkl'))

feature_names = data['feature_names']
X_train_df = pd.DataFrame(data['X_res'], columns=feature_names)
X_test_df  = pd.DataFrame(data['X_test'], columns=feature_names)

print("Running LinearExplainer...")
explainer = shap.LinearExplainer(model, X_train_df)
shap_vals = explainer.shap_values(X_test_df)

print('type:', type(shap_vals))
print('shape:', getattr(shap_vals, 'shape', [s.shape for s in shap_vals]))
print("Done — no errors")