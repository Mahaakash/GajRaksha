import shap
import xgboost as xgb
import numpy as np
import pandas as pd

print("SHAP version:", shap.__version__)
print("XGBoost version:", xgb.__version__)

X_dummy = pd.DataFrame(np.random.randn(100, 8), columns=[f"feat_{i}" for i in range(8)])
y_dummy = np.random.randint(0, 2, 100)

clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
clf.fit(X_dummy, y_dummy)

explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_dummy)
print("TreeExplainer works! SHAP values shape:", shap_vals.shape)

