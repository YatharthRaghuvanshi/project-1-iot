import pandas as pd

from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/context_updated.csv")

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

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    print(f"\nFold {fold+1}")

    print("Before SMOTE")

    print(y_train.value_counts())
    print(X.dtypes)

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nAfter SMOTE")

    print(pd.Series(y_resampled).value_counts())