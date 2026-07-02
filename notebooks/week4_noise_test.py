import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# Load dataset
df = pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/context_updated.csv")

# Drop non-feature columns
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
y = df["Machine failure"]

# Clean feature names
X.columns = (
    X.columns
    .str.replace("[^A-Za-z0-9_]", "_", regex=True)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

clean_pred = model.predict(X_test)
clean_f1 = f1_score(y_test, clean_pred, average="macro")

print("Clean Macro F1:", clean_f1)

noise_levels = [0.05,0.15,0.30]

data=[]

for noise in noise_levels:

    noisy = X_test + np.random.normal(
        0,
        noise,
        X_test.shape
    )

    pred = model.predict(noisy)

    f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    drop = clean_f1-f1

    data.append(
        [noise,f1,drop]
    )

data = pd.DataFrame(
    data,
    columns=[
        "Noise Level",
        "Macro F1",
        "F1 Drop"
    ]
)

print(data)

data.to_csv(
    "C:/Users/ASUS/Downloads/project-1-iot/data/noise_results.csv",
    index=False
)