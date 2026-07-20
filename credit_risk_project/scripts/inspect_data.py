import pandas as pd

# nrows=5 means "only read the first 5 rows" - loads almost instantly
# even though the full file is enormous, letting us see the structure safely first
df_sample = pd.read_csv("../data/raw/accepted_2007_to_2018Q4.csv", nrows=5)

print("Number of columns:", df_sample.shape[1])
print("\nColumn names:")
print(df_sample.columns.tolist())