import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier

df = pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/context_updated.csv")
X = df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_enc"
    ]
]
y = df["Machine failure"]

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("lgbm", LGBMClassifier(random_state=42))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
print (X.dtypes)
scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=[
        "f1_macro",
        "precision_macro",
        "recall_macro"
    ]
)

print("Macro F1 Scores")
print(scores["test_f1_macro"])

print("\nMean F1")
print(scores["test_f1_macro"].mean())

print("\nStd Dev")
print(scores["test_f1_macro"].std())

print("\nPrecision")
print(scores["test_precision_macro"].mean())

print("\nRecall")
print(scores["test_recall_macro"].mean())