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

X = numeric_df.drop(columns=["Machine_failure"])
y = numeric_df["Machine_failure"]

model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X, y)

print("Optimized model trained successfully.")