# Model Card: LendingClub Loan Default Prediction

## Overview
A binary classification model predicting the probability that a loan will default (charge off), trained on LendingClub's accepted loan data (2007–2018). Built to demonstrate an end-to-end credit risk workflow: data cleaning, leakage-aware feature engineering, class-imbalance handling, model comparison, explainability, probability calibration, and deployment.

## Intended Use
Educational portfolio project demonstrating credit risk modeling techniques. **Not intended for actual lending decisions.** A real deployment would require additional validation (see Limitations).

## Data
- Source: LendingClub accepted loans, 2007–2018 (Kaggle)
- Training rows: 1,345,350 resolved loans (Fully Paid, Charged Off, or Default)
- Excluded: loans with ambiguous/unresolved status (Current, Late, In Grace Period, discontinued credit policy loans)
- Target: `default` (1 = Charged Off/Default, 0 = Fully Paid)
- Base rate: ~19.97% default

## Features
26 features known at loan application time, including loan terms (amount, rate, term, grade), borrower financials (income, DTI, employment length), and credit history (FICO, open accounts, delinquencies, credit history length). One-hot encoded to 92 total columns after processing.

**Deliberately excluded (data leakage):** payment history, recovery amounts, hardship/settlement flags, and any field only populated after loan issuance or after a loan enters distress — these would not be available at the time a real lending decision is made.

## Models Trained
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.663 | 0.326 | 0.644 | 0.433 | 0.715 |
| XGBoost | 0.652 | 0.323 | 0.682 | 0.439 | 0.724 |

XGBoost was selected as the primary model for a modest recall improvement (catches ~4 percentage points more actual defaulters), though the gap over the simpler, fully interpretable Logistic Regression baseline is small — worth weighing against XGBoost's reduced transparency in a real regulatory context.

## Explainability
SHAP (TreeExplainer) used for both global feature importance and individual prediction explanations. Top global drivers: sub-grade, grade, loan term, DTI, FICO score. Notably, a missing-employment-length flag was found to independently signal elevated risk, validating the decision to encode missingness explicitly rather than impute it away.

## Probability Calibration
The trained XGBoost model, fit with `scale_pos_weight` to address class imbalance, produced systematically inflated probabilities (average predicted default rate of 45.3% against a true rate of 19.97%). Isotonic calibration (fit on a held-out calibration split, evaluated on a separate final split) corrected this to 19.94% average predicted probability, improving Brier score from 0.2124 to 0.1428 (~33% improvement).

**Known limitation:** isotonic calibration showed reduced granularity at the high-risk tail — two synthetic test profiles with clearly different risk factors (one strictly worse than the other on every input) produced identical calibrated probabilities. This is a known property of isotonic regression when calibration data is sparse in a given score region, and would need a larger calibration set or an alternative method (e.g. Platt scaling) to fully resolve in production.

## Limitations
- **No out-of-time validation:** train/test split is random, not time-based. A real deployment should validate on more recent loans than it trained on, since default patterns shift with economic conditions.
- **No fairness/bias audit:** the model has not been tested for disparate impact across protected characteristics (directly or via proxies like geography).
- **No monitoring plan:** a production model requires ongoing performance tracking (e.g. Population Stability Index) as the applicant population and economy evolve.
- **Calibration tail limitation:** as noted above, high-risk-tail probabilities lack fine granularity.
- **Static thresholds:** the app's Low/Moderate/High risk cutoffs (15%/30%) are illustrative, not derived from an actual cost-benefit analysis a bank would perform.

## Files
- `scripts/clean_data.py`, `feature_engineering.py` — data preparation
- `scripts/train_model.py` — model training and evaluation
- `scripts/explain_model.py` — SHAP analysis
- `scripts/calibrate_model.py` — probability calibration
- `app/streamlit_app.py` — interactive prediction tool