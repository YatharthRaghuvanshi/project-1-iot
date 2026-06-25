import pandas as pd
from lightgbm import LGBMClassifier


df = pd.read_csv("data/context_updated.csv")


df.columns = (
    df.columns
      .str.replace("[", "", regex=False)
      .str.replace("]", "", regex=False)
      .str.replace(" ", "_", regex=False)
)


numeric_df = df.select_dtypes(include=["int64", "float64"])

y = numeric_df["Machine_failure"]

X = numeric_df.drop(columns=["Machine_failure"])

print("Features being used:")
print(X.columns.tolist())

print("\nDataset Shape:")
print(X.shape)

model = LGBMClassifier(
    random_state=42,
    verbose=-1
)

model.fit(X, y)

print("\nModel trained successfully!")
print("Number of features:", len(X.columns))