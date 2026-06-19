import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv("data/context_updated.csv")

base_features = [
    "Type_enc",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

extended_features = (
    base_features
    + [
        "ambient_temp_C",
        "factory_load_pct"
    ]
)

target = "Machine failure"


def evaluate(features):

    X = df[features]
    y = df[target]

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

    return f1_score(
        y_test,
        preds,
        average="macro"
    )


base_f1 = evaluate(base_features)

context_f1 = evaluate(
    extended_features
)

print("Base F1:", base_f1)

print("Context F1:", context_f1)