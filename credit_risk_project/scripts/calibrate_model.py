import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.frozen import FrozenEstimator

# Reproduce the exact same split used during training
df = pd.read_csv("../data/processed/model_ready_data.csv")
X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split test data further: half for calibration, half for final honest evaluation
X_calib, X_final, y_calib, y_final = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)

print("Calibration set:", X_calib.shape)
print("Final evaluation set:", X_final.shape)

# Load the already-trained model
xgb_model = joblib.load("../app/xgb_model.pkl")

# Check RAW (uncalibrated) probabilities first, for comparison
raw_proba = xgb_model.predict_proba(X_final)[:, 1]
print("\nRaw model - average predicted probability:", raw_proba.mean().round(4))
print("Actual default rate in this set:          ", y_final.mean().round(4))
print("Raw Brier score (lower is better):", round(brier_score_loss(y_final, raw_proba), 4))

# Fit a calibrator on top of the already-trained model.
# method="isotonic" is a flexible, non-parametric calibration approach -
# well suited to large datasets like ours (100K+ calibration rows),
# where it can learn a more accurate mapping than the simpler "sigmoid" method.
# cv="prefit" tells it: don't retrain the underlying model, it's already trained -
# just learn a calibration mapping on top of its existing outputs.
calibrated_model = CalibratedClassifierCV(FrozenEstimator(xgb_model), method="isotonic")
calibrated_model.fit(X_calib, y_calib)

# Now check calibrated probabilities on the SAME final evaluation set as before
calibrated_proba = calibrated_model.predict_proba(X_final)[:, 1]

print("\n--- AFTER CALIBRATION ---")
print("Calibrated - average predicted probability:", round(calibrated_proba.mean(), 4))
print("Actual default rate in this set:            ", round(y_final.mean(), 4))
print("Calibrated Brier score (lower is better):", round(brier_score_loss(y_final, calibrated_proba), 4))

# Save the calibrated model - this replaces the raw model in the Streamlit app
joblib.dump(calibrated_model, "../app/xgb_model_calibrated.pkl")
print("\nSaved calibrated model to app/xgb_model_calibrated.pkl")