import pandas as pd

df = pd.read_csv("data/context_updated.csv")

# Clean column names
df.columns = (
    df.columns
      .str.replace("[", "", regex=False)
      .str.replace("]", "", regex=False)
      .str.replace(" ", "_", regex=False)
)

numeric_df = df.select_dtypes(include=["int64", "float64"])

X = numeric_df.drop(columns=["Machine_failure"])
y = numeric_df["Machine_failure"]

print("Dataset Shape:", X.shape)
print("Target Distribution:")
print(y.value_counts())