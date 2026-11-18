import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import load_data, prepare_data


DATA_PATH = "data/Titanic-Dataset.csv"

MODELS = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
}


# Load and prepare data
df = load_data(DATA_PATH)
X, y = prepare_data(df)

# Use the exact same split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


results = []


for model_name, model_path in MODELS.items():

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    })


results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
        }
    )
)


# Select model based on F1 Score
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBest Model")
print("----------")
print(f"Model    : {best_model['Model']}")
print(f"F1 Score : {best_model['F1 Score']:.4f}")