import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/processed/model_ready_data.csv")

print("Full dataset shape:", df.shape)

# Separate features (X) from target (y)
X = df.drop(columns=["default"])
y = df["default"]

# Split: 80% train, 20% test
# stratify=y ensures both sets keep the same ~20% default rate,
# critical for an imbalanced dataset like this one
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Logistic Regression is sensitive to feature scale (e.g. loan_amnt in thousands
# vs dti as a small decimal) - scaling puts all features on comparable footing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# class_weight="balanced" automatically adjusts for our ~20% default rate,
# telling the model to pay proportionally more attention to the minority class
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

print("\nLogistic Regression training complete.")

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nDefault rate in training set:", y_train.mean().round(4))
print("Default rate in test set:", y_test.mean().round(4))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Predict on the held-out test set
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]  # probability of default

print("\n--- LOGISTIC REGRESSION PERFORMANCE ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))