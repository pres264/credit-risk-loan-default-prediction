import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the saved model and data
xgb_model = joblib.load("../app/xgb_model.pkl")
feature_names = joblib.load("../app/feature_names.pkl")

df = pd.read_csv("../data/processed/model_ready_data.csv")
X = df.drop(columns=["default"])

# SHAP can be slow on huge datasets - use a representative sample, not all 1.3M rows
X_sample = X.sample(n=5000, random_state=42)

print("Calculating SHAP values on a sample of", len(X_sample), "rows...")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated.")

# Global summary plot: which features matter most, and how they push predictions
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Summary: What Drives Default Risk", fontsize=14)
plt.tight_layout()
plt.savefig("../docs/shap_summary_plot.png", dpi=150)
plt.show()

print("\nSaved global summary plot to docs/shap_summary_plot.png")

# Pick one specific loan to explain in detail
sample_idx = 0
single_prediction = xgb_model.predict_proba(X_sample.iloc[[sample_idx]])[:, 1][0]

print(f"\nExplaining prediction for one applicant:")
print(f"Predicted default probability: {single_prediction:.2%}")

plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[sample_idx],
        feature_names=X_sample.columns.tolist()
    ),
    show=False
)
plt.title("Why This Applicant Was Scored This Way", fontsize=12)
plt.tight_layout()
plt.savefig("../docs/shap_waterfall_example.png", dpi=150)
plt.show()

print("Saved individual explanation to docs/shap_waterfall_example.png")