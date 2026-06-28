import pandas as pd
import joblib

from lightgbm import LGBMClassifier

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

model = LGBMClassifier(
    random_state=42,
    verbose=-1
)

model.fit(X, y)

joblib.dump(model, "models/lightgbm_model.pkl")

print("Model saved successfully!")