# Loan Default Risk Prediction

An end-to-end machine learning project predicting loan default risk from LendingClub data (2007–2018), built with explainability and probability calibration as first-class concerns — not just an afterthought.

## What makes this different from a typical ML portfolio project

- **Leakage-aware feature selection** — deliberately excludes any field only known after a loan is issued or after it enters distress
- **Two models compared honestly** — an interpretable Logistic Regression baseline vs. XGBoost, with an explicit discussion of the accuracy/transparency tradeoff, not just "picked the best score"
- **SHAP explainability** — both portfolio-wide feature importance and individual, per-applicant explanations
- **Probability calibration** — identified that class-weighting had inflated predicted probabilities (45% average predicted vs. 20% true default rate), fixed with isotonic calibration, and documented the resulting tail-granularity limitation honestly rather than hiding it
- **Deployed, interactive app** — not just a notebook; a working Streamlit tool with live predictions and live SHAP explanations

## Key Results

| Model | ROC-AUC | Recall |
|---|---|---|
| Logistic Regression | 0.715 | 0.644 |
| XGBoost | 0.724 | 0.682 |

Post-calibration Brier score improved ~33% (0.2124 → 0.1428), bringing average predicted probability (19.94%) in line with the true default rate (19.97%).

See [`docs/model_card.md`](docs/model_card.md) for full methodology, feature list, and documented limitations.

## Project Structure

```
credit_risk_project/
├── data/
│   ├── raw/              # LendingClub CSV (not tracked - see setup)
│   └── processed/        # Cleaned, model-ready data (not tracked - regenerate via scripts)
├── scripts/
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── explain_model.py
│   ├── calibrate_model.py
│   └── save_defaults.py
├── app/
│   └── streamlit_app.py
├── docs/
│   └── model_card.md
└── README.md
```

## Setup

```bash
# 1. Download accepted_2007_to_2018Q4.csv from Kaggle ("All Lending Club loan data" by Nathan George)
#    and place it in data/raw/

pip install pandas scikit-learn xgboost shap imbalanced-learn streamlit joblib

cd scripts
python clean_data.py
python feature_engineering.py
python train_model.py
python calibrate_model.py
python save_defaults.py

cd ../app
streamlit run streamlit_app.py
```

## Tools Used

Python (Pandas, scikit-learn, XGBoost, SHAP, imbalanced-learn) · Streamlit · Git/GitHub

## Author

Presley Oluoch — [LinkedIn](https://linkedin.com/in/presley-oluoch-992195279) · [GitHub](https://github.com/pres264) · [Portfolio](https://your-biz-beacon.lovable.app)