import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/cleaned_ai4i.csv")

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

for col in sensor_cols:

    plt.figure(figsize=(8,5))

    sns.boxplot(
        data=df,
        x="Machine failure",
        y=col
    )

    plt.title(col)

    plt.show()