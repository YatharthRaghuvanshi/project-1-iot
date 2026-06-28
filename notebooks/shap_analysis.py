import pandas as pd
import shap

from lightgbm import LGBMClassifier

df = pd.read_csv("data/context_updated.csv")

df.columns = (
    df.columns
      .str.replace("[", "", regex=False)
      .str.replace("]", "", regex=False)
      .str.replace(" ", "_", regex=False)
)

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

X = numeric_df.drop(
    columns=["Machine_failure"]
)

y = numeric_df["Machine_failure"]

model = LGBMClassifier(
    random_state=42,
    verbose=-1
)

model.fit(X, y)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

print("SHAP values generated successfully")