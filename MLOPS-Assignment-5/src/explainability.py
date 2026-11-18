
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
df = pd.read_csv(ROOT / "data" / "Titanic-Dataset.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Survived"])

features = ["Pclass", "Sex", "Age", "SibSp",
            "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"].astype(int)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load(ROOT / "models" / "RandomForest.joblib")
result = permutation_importance(
    model, X_test, y_test, n_repeats=10,
    random_state=42, scoring="f1"
)

importance = pd.DataFrame({
    "feature": features,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean")

importance.to_csv(RESULTS / "feature_importance.csv", index=False)
importance.plot.barh(
    x="feature", y="importance_mean", xerr="importance_std",
    legend=False, title="Random Forest: Permutation Importance"
)
plt.xlabel("Decrease in F1 when feature is shuffled")
plt.tight_layout()
plt.savefig(RESULTS / "feature_importance.png", dpi=160)
print(importance.to_string(index=False))
print("Saved results/feature_importance.png")
