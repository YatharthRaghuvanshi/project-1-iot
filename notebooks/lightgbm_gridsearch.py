import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV

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

model = LGBMClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid.fit(X, y)

print("Best Parameters:")
print(grid.best_params_)

print("\nBest Macro F1:")
print(grid.best_score_)