
from pathlib import Path

import pandas as pd
from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate,
    false_positive_rate
)
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
df = pd.read_csv(RESULTS / "test_predictions.csv")

y = df["actual"]
groups = df["sex"]

for model_name in ["LogisticRegression", "RandomForest"]:
    pred = df[f"{model_name}_prediction"]

    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "recall": true_positive_rate,
            "false_positive_rate": false_positive_rate
        },
        y_true=y,
        y_pred=pred,
        sensitive_features=groups
    )

    print(f"\n{model_name}: metrics by sex")
    print(frame.by_group.round(4).to_string())

    frame.by_group.to_csv(
        RESULTS / f"fairness_{model_name}.csv"
    )
    print("Metric differences (max - min):")
    print(frame.difference(method="between_groups").round(4))
