import pandas as pd
import joblib

df = pd.read_csv("../data/processed/model_ready_data.csv")
X = df.drop(columns=["default"])

# Use median for numeric stability, and this becomes our "typical applicant" baseline
default_values = X.median().to_dict()

joblib.dump(default_values, "../app/default_values.pkl")
print("Saved default values for", len(default_values), "features.")
print("\nSample defaults:")
for k, v in list(default_values.items())[:10]:
    print(f"  {k}: {v}")