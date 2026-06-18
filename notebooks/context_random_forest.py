import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

import sys
import os

sys.path.append(os.path.abspath("."))

from src.feature_sets import extended_features

df = pd.read_csv("data/cleaned_ai4i.csv")

print(df.columns.tolist())
X = df[extended_features]

y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print(
    "Precision:",
    precision_score(y_test, preds)
)

print(
    "Recall:",
    recall_score(y_test, preds)
)

print(
    "Macro F1:",
    f1_score(
        y_test,
        preds,
        average="macro"
    )
)