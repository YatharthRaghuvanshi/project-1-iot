from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier

model_pipeline = Pipeline(

    steps=[

        (
            "smote",
            SMOTE(random_state=42)
        ),

        (
            "lgbm",
            LGBMClassifier(
                random_state=42
            )
        )
    ]
)

print(model_pipeline)