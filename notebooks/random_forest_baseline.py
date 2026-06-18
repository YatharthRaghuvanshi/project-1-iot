import sys
import os

sys.path.append(os.path.abspath("."))

from src.feature_sets import base_features

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from src.feature_sets import base_features

df = pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/rolling_ai4i.csv")

X = df[base_features]

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

precision = precision_score(y_test, preds)

recall = recall_score(y_test, preds)

macro_f1 = f1_score(
    y_test,
    preds,
    average="macro"
)

print("Precision:", precision)
print("Recall:", recall)
print("Macro F1:", macro_f1)