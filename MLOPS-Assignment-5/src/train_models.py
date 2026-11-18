
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Titanic-Dataset.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()

required = {"Survived", "Pclass", "Sex", "Age", "SibSp",
            "Parch", "Fare", "Embarked"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["Survived"]).copy()
df["Survived"] = df["Survived"].astype(int)

features = ["Pclass", "Sex", "Age", "SibSp",
            "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

# Preserve group labels for the later fairness assessment.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical = ["Sex", "Embarked"]

def make_preprocessor(scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

models = {
    "LogisticRegression": Pipeline([
        ("preprocess", make_preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=5,
            random_state=42, class_weight=None
        ))
    ])
}

metrics = []
predictions = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "sex": X_test["Sex"].reset_index(drop=True),
    "age": X_test["Age"].reset_index(drop=True)
})

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    })

    predictions[f"{name}_prediction"] = pred
    joblib.dump(model, MODELS / f"{name}.joblib")

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(RESULTS / "model_metrics.csv", index=False)
predictions.to_csv(RESULTS / "test_predictions.csv", index=False)

with open(RESULTS / "experiment_config.json", "w") as f:
    json.dump({
        "dataset": DATA.name,
        "random_state": 42,
        "test_size": 0.2,
        "models": list(models),
        "features": features
    }, f, indent=2)

print(f"Dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
print("\nModel evaluation:")
print(metrics_df.round(4).to_string(index=False))
print(f"\nSaved models to: {MODELS}")
print(f"Saved results to: {RESULTS}")
