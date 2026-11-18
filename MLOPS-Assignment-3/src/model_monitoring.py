import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocessing import load_data, prepare_data


DATA_PATH = "data/Titanic-Dataset.csv"
MODEL_PATH = "models/decision_tree.pkl"


def evaluate_model():

    # Load data
    df = load_data(DATA_PATH)

    # Prepare features and target
    X, y = prepare_data(df)

    # Same split used during model training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    return metrics


if __name__ == "__main__":

    metrics = evaluate_model()

    print("\nModel Performance Monitoring")
    print("============================")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")