import pandas as pd
from src.feature_sets import base_features

df = pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/cleaned_ai4i.csv")

missing = [col for col in base_features if col not in df.columns]

print("Missing Features:", missing)
print("Total Features:", len(base_features))