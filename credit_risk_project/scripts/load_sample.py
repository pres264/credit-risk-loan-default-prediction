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

print("Shape:", df.shape)
print("\nLoan status value counts:")
print(df["loan_status"].value_counts())