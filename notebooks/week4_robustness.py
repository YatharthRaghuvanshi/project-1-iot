import numpy as np

def inject_noise(X_test, noise_level):
    noisy = X_test.copy()

    numeric_cols = noisy.select_dtypes(include=["number"]).columns

    noisy[numeric_cols] = (
        noisy[numeric_cols]
        + np.random.normal(
            0,
            noise_level,
            noisy[numeric_cols].shape
        )
    )

    return noisy