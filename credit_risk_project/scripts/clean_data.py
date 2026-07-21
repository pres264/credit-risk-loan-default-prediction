import pandas as pd

keep_columns = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "addr_state", "loan_status"
]

df = pd.read_csv(
    "../data/raw/accepted_2007_to_2018Q4.csv",
    usecols=keep_columns
)
# Keeping only resolved outcomes - excluding anything still in progress or ambiguous
resolved_statuses = ["Fully Paid", "Charged Off", "Default"]
df = df[df["loan_status"].isin(resolved_statuses)].copy()

# Building binary target: 1 = defaulted, 0 = fully paid
df["default"] = df["loan_status"].apply(
    lambda x: 1 if x in ["Charged Off", "Default"] else 0
)
# Clean 'term': "36 months" -> 36 (integer)
df["term"] = df["term"].str.replace(" months", "", regex=False).astype(int)

# Clean 'emp_length': text categories -> numeric years
emp_length_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10
}
df["emp_length"] = df["emp_length"].map(emp_length_map)

# Parse dates properly
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")

# Feature engineering: how long has this person had credit, at loan issue time?
# This is more useful to a model than a raw date - it's the actual credit history length
df["credit_history_years"] = (
    (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
)

# emp_length: missing likely means unemployed/undisclosed - flag it, then fill with 0
df["emp_length_missing"] = df["emp_length"].isnull().astype(int)
df["emp_length"] = df["emp_length"].fillna(0)

# mort_acc: missing likely means no mortgage accounts exist
df["mort_acc"] = df["mort_acc"].fillna(0)

# Small-gap columns: fill with median (robust to outliers, unlike mean)
for col in ["revol_util", "pub_rec_bankruptcies", "dti", "inq_last_6mths"]:
    df[col] = df[col].fillna(df[col].median())

print("\nRemaining missing values:", df.isnull().sum().sum())

# Save the cleaned dataset
df.to_csv("../data/processed/cleaned_loan_data.csv", index=False)
print("Saved cleaned data to data/processed/cleaned_loan_data.csv")
print("Final shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nRows after filtering to resolved loans only:", df.shape[0])
print("\nDefault rate:")
print(df["default"].value_counts(normalize=True))