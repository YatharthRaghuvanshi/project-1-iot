import pandas as pd

from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("data/context_updated.csv")

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(y.value_counts(normalize=True))

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"\nFold {fold+1}")

    print("Train:")
    print(y_train.value_counts(normalize=True))

    print("Test:")
    print(y_test.value_counts(normalize=True))