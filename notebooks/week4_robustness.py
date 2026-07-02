import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)

from lightgbm import LGBMClassifier

df = pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/context_updated.csv")


drop_cols = [
    "UDI",
    "Product ID",
    "Type",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

X = df.drop(columns=drop_cols)

X.columns = (
    X.columns
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace(" ", "_")
    .str.replace("/", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)
y = df["Machine failure"]
print(X.columns.tolist())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = LGBMClassifier(random_state=42)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]


precision, recall, thresholds = precision_recall_curve(
    y_test,
    probs
)

pr_auc = auc(recall, precision)

print("=" * 40)
print("Precision-Recall AUC:", round(pr_auc, 4))
print("=" * 40)

plt.figure(figsize=(8,6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.grid(True)
plt.show()


print("\nThreshold Sweep\n")

results = []

for threshold in np.arange(0.10, 0.95, 0.05):

    preds = (probs >= threshold).astype(int)

    precision_val = precision_score(
        y_test,
        preds,
        zero_division=0
    )

    recall_val = recall_score(
        y_test,
        preds,
        zero_division=0
    )

    f1_val = f1_score(
        y_test,
        preds,
        zero_division=0
    )

    results.append([
        threshold,
        precision_val,
        recall_val,
        f1_val
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1"
    ]
)

print(results_df)

best = results_df.loc[results_df["F1"].idxmax()]

print("\nBest Threshold")
print(best)


def inject_noise(X, noise_level):

    noisy = X.copy()

    numeric_cols = noisy.select_dtypes(include=np.number).columns

    noise = np.random.normal(
        0,
        noise_level,
        noisy[numeric_cols].shape
    )

    noisy[numeric_cols] = noisy[numeric_cols] + noise

    return noisy


X_noisy = inject_noise(X_test, 0.10)

noisy_probs = model.predict_proba(X_noisy)[:, 1]

print("\nPrediction on noisy data successful.")