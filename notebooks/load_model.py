import pandas as pd
import joblib

df = pd.read_csv("data/context_updated.csv")

df.columns = (
    df.columns
      .str.replace("[", "", regex=False)
      .str.replace("]", "", regex=False)
      .str.replace(" ", "_", regex=False)
)

numeric_df = df.select_dtypes(include=["int64", "float64"])

X = numeric_df.drop(columns=["Machine_failure"])

model = joblib.load("models/lightgbm_model.pkl")

predictions = model.predict(X.head(10))

print("Predictions:")

print(predictions)