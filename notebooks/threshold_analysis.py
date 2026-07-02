import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
precision_score,
recall_score,
f1_score
)

from sklearn.model_selection import train_test_split

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

prob=model.predict_proba(X_test)[:,1]

thresholds=np.arange(0.1,0.95,0.05)

precision=[]
recall=[]
f1=[]

for t in thresholds:

    pred=(prob>=t).astype(int)

    precision.append(
        precision_score(y_test,pred)
    )

    recall.append(
        recall_score(y_test,pred)
    )

    f1.append(
        f1_score(y_test,pred)
    )

best=np.argmax(f1)

plt.figure(figsize=(12,4))

plt.plot(thresholds,f1,label="F1")

plt.plot(thresholds,precision,label="Precision")

plt.plot(thresholds,recall,label="Recall")

plt.scatter(
thresholds[best],
f1[best],
color="red",
s=100
)

plt.legend()

plt.xlabel("Threshold")

plt.savefig("C:/Users/ASUS/Downloads/project-1-iot/outputs/threshold_analysis.png")

plt.show()

print("Best Threshold =",thresholds[best])