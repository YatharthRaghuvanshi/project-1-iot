import pandas as pd

df = pd.read_csv("data/context_updated.csv")

# Clean column names
df.columns = (
    df.columns
      .str.replace("[", "", regex=False)
      .str.replace("]", "", regex=False)
      .str.replace(" ", "_", regex=False)
)

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())