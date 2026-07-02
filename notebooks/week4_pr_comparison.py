import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from lightgbm import LGBMClassifier

df=pd.read_csv("C:/Users/ASUS/Downloads/project-1-iot/data/context_updated.csv")

drop_cols=[
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

X=df.drop(columns=drop_cols)
y=df["Machine failure"]

X.columns=X.columns.str.replace("[^A-Za-z0-9_]","_",regex=True)

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

model=LGBMClassifier(random_state=42)

model.fit(X_train,y_train)

levels={
"Clean":0,
"Low":0.05,
"Medium":0.15,
"High":0.30
}

plt.figure(figsize=(8,6))

for name,std in levels.items():

    noisy=X_test+np.random.normal(
        0,
        std,
        X_test.shape
    )

    probs=model.predict_proba(noisy)[:,1]

    precision,recall,_=precision_recall_curve(
        y_test,
        probs
    )

    plt.plot(
        recall,
        precision,
        label=name
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve Comparison")

plt.legend()

plt.savefig(
"C:/Users/ASUS/Downloads/project-1-iot/outputs/pr_curves_comparison.png",
dpi=300
)

plt.show()