import pandas as pd

df = pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/context_updated.csv")

print(df[
    [
        "ambient_temp_C",
        "factory_load_pct"
    ]
].describe())
import matplotlib.pyplot as plt

df["ambient_temp_C"].hist()

plt.title(
    "Ambient Temperature Distribution"
)

plt.show()
df["factory_load_pct"].hist()

plt.title(
    "Factory Load Distribution"
)

plt.show()