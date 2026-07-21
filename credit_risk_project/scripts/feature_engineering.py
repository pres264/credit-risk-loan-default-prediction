import pandas as pd

df = pd.read_csv("../data/processed/cleaned_loan_data.csv")

print("Shape before encoding:", df.shape)

# Ordinal encode grade (A=1 best ... G=7 worst)
grade_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["grade_encoded"] = df["grade"].map(grade_order)

# Ordinal encode sub_grade (A1=1 best ... G5=35 worst)
grades = ["A", "B", "C", "D", "E", "F", "G"]
sub_grade_order = {}
rank = 1
for letter in grades:
    for number in range(1, 6):
        sub_grade_order[f"{letter}{number}"] = rank
        rank += 1

df["sub_grade_encoded"] = df["sub_grade"].map(sub_grade_order)

# One-hot encode nominal (unordered) categorical columns
nominal_cols = ["home_ownership", "verification_status", "purpose", "addr_state"]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

print("\nShape after one-hot encoding:", df.shape)

# Drop original text/date columns now replaced by encoded/engineered versions
columns_to_drop = ["grade", "sub_grade", "loan_status", "issue_d", "earliest_cr_line"]
df = df.drop(columns=columns_to_drop)

print("\nFinal shape:", df.shape)
print("\nFinal columns:")
print(df.columns.tolist())

# Save the fully encoded, model-ready dataset
df.to_csv("../data/processed/model_ready_data.csv", index=False)
print("\nSaved model-ready data to data/processed/model_ready_data.csv")