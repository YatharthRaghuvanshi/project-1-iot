import pandas as pd

df = pd.read_csv("data/context_updated.csv")

X = df.drop(columns=["Machine failure"])

y = df["Machine failure"]

print("Feature Matrix Shape")

print(X.shape)

print("\nTarget Shape")

print(y.shape)

print("\nMissing Values")

print(df.isnull().sum().sum())

print("\nClass Distribution")

print(y.value_counts())