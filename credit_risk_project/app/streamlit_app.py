import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def calculate_installment(loan_amnt, int_rate, term):
    monthly_rate = (int_rate / 100) / 12
    if monthly_rate == 0:
        return loan_amnt / term
    installment = loan_amnt * monthly_rate / (1 - (1 + monthly_rate) ** -term)
    return round(installment, 2)


# Load everything we saved earlier - paths built relative to this script's
# own location, so it works whether run locally or from Streamlit Cloud's
# repo root
xgb_model_raw = joblib.load(os.path.join(APP_DIR, "xgb_model.pkl"))            # for SHAP explanations
xgb_model = joblib.load(os.path.join(APP_DIR, "xgb_model_calibrated.pkl"))     # for the actual probability shown
feature_names = joblib.load(os.path.join(APP_DIR, "feature_names.pkl"))
default_values = joblib.load(os.path.join(APP_DIR, "default_values.pkl"))

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Loan Default Risk Predictor")
st.write("Enter applicant details to predict default risk, with an explanation of the decision.")

# --- Input form ---
st.header("Applicant Information")

loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=12000, step=500)
term = st.selectbox("Loan Term (months)", [36, 60])
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.7)
annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=1000)
dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 17.6)
fico = st.slider("FICO Score", 300, 850, 690)
grade_encoded = st.selectbox("Loan Grade", options=[1,2,3,4,5,6,7],
                              format_func=lambda x: ["A","B","C","D","E","F","G"][x-1])
emp_length = st.slider("Employment Length (years)", 0, 10, 6)
home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])

predict_button = st.button("Predict Default Risk")

if predict_button:
    # Start with the "typical applicant" defaults for every feature
    input_data = default_values.copy()

    # Override with the specific values the user actually entered
    input_data["loan_amnt"] = loan_amnt
    input_data["term"] = term
    input_data["int_rate"] = int_rate
    input_data["installment"] = calculate_installment(loan_amnt, int_rate, term)
    input_data["annual_inc"] = annual_inc
    input_data["dti"] = dti
    input_data["fico_range_low"] = fico
    input_data["fico_range_high"] = fico + 4  # LendingClub always reports a small range
    input_data["grade_encoded"] = grade_encoded
    input_data["emp_length"] = emp_length

    # Handle home_ownership one-hot encoding manually
    input_data["home_ownership_RENT"] = 1 if home_ownership == "RENT" else 0
    input_data["home_ownership_MORTGAGE"] = 1 if home_ownership == "MORTGAGE" else 0
    # Note: OWN is the "dropped" baseline category from one-hot encoding,
    # so it's represented by both RENT and MORTGAGE being 0 - no explicit column needed

    # Build a single-row dataframe matching the exact column order the model expects
    input_df = pd.DataFrame([input_data])[feature_names]

    # Get prediction
    probability = xgb_model.predict_proba(input_df)[:, 1][0]

    st.header("Prediction Result")
    st.metric("Default Probability", f"{probability:.1%}")

    if probability < 0.15:
        st.success("Low Risk")
    elif probability < 0.30:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")

    st.header("Why This Prediction?")

    explainer = shap.TreeExplainer(xgb_model_raw)
    shap_values_single = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)

    with st.expander("Debug: show input values"):
        st.write(input_df)