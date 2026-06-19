import matplotlib.pyplot as plt

models = [
    "Baseline",
    "Context"
]

scores = [
    0.82,
    0.86
]

plt.figure(figsize=(6,4))

plt.bar(
    models,
    scores
)

plt.ylabel("Macro F1")

plt.title(
    "Ablation Study Results"
)

plt.savefig(
    "outputs/ablation_results.png"
)

plt.show()