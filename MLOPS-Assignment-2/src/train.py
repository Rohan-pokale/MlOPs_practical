import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def load_and_preprocess_data():
    """Load and preprocess the Titanic dataset."""

    df = pd.read_csv("data/Titanic-Dataset.csv")

    print("Dataset Shape:", df.shape)

    # Drop unnecessary columns
    df.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        inplace=True,
    )

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    # Encode categorical columns
    encoder = LabelEncoder()

    for column in ["Sex", "Embarked"]:
        df[column] = encoder.fit_transform(df[column])

    # Separate features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test


def train_and_log_model(
    model,
    model_name,
    run_name,
    model_file,
    parameters,
):
    """Train a model and log the experiment in MLflow."""

    with mlflow.start_run(run_name=run_name):

        # Log model parameters
        mlflow.log_param("model", model_name)

        for parameter_name, parameter_value in parameters.items():
            mlflow.log_param(
                parameter_name,
                parameter_value,
            )

        # Log reproducibility information
        mlflow.set_tag(
            "python_version",
            sys.version.split()[0],
        )

        mlflow.set_tag(
            "scikit_learn_version",
            sklearn.__version__,
        )

        mlflow.set_tag(
            "dataset",
            "Titanic-Dataset.csv",
        )

        mlflow.set_tag(
            "train_test_split",
            "80/20",
        )

        # Train model
        model.fit(X_train, y_train)

        # Generate predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
        )

        recall = recall_score(
            y_test,
            predictions,
        )

        f1 = f1_score(
            y_test,
            predictions,
        )

        # Print results
        print(f"\n{model_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Create and save confusion matrix
        cm = confusion_matrix(
            y_test,
            predictions,
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
        )

        disp.plot()

        import matplotlib.pyplot as plt

        plt.title(
            f"{model_name} - Confusion Matrix"
        )

        plt.tight_layout()

        confusion_matrix_path = (
            "confusion_matrix.png"
        )

        plt.savefig(confusion_matrix_path)
        plt.close()

        mlflow.log_artifact(
            confusion_matrix_path
        )

        # Classification report
        report = classification_report(
            y_test,
            predictions,
        )

        print("\nClassification Report:")
        print(report)

        report_path = "classification_report.txt"

        with open(
            report_path,
            "w",
        ) as file:
            file.write(report)

        mlflow.log_artifact(report_path)

        # Save local model
        joblib.dump(
            model,
            model_file,
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            name=model_name.replace(
                " ",
                "_",
            ).lower() + "_model",
        )

        print(
            f"Model saved to {model_file}"
        )


def main():

    # Create/use MLflow experiment
    mlflow.set_experiment(
        "Titanic_Classification"
    )

    # Load and preprocess data
    global X_train, X_test, y_train, y_test

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_and_preprocess_data()

    # -----------------------------
    # 1. Random Forest
    # -----------------------------

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    train_and_log_model(
        model=random_forest,
        model_name="Random Forest",
        run_name="Random_Forest_Baseline",
        model_file="models/titanic_model.pkl",
        parameters={
            "n_estimators": 100,
            "random_state": 42,
            "test_size": 0.2,
        },
    )

    # -----------------------------
    # 2. Decision Tree
    # -----------------------------

    decision_tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )

    train_and_log_model(
        model=decision_tree,
        model_name="Decision Tree",
        run_name="Decision_Tree",
        model_file="models/titanic_decision_tree.pkl",
        parameters={
            "max_depth": 5,
            "random_state": 42,
            "test_size": 0.2,
        },
    )

    # -----------------------------
    # 3. Logistic Regression
    # -----------------------------

    logistic_regression = LogisticRegression(
        max_iter=200,
        random_state=42,
    )

    train_and_log_model(
        model=logistic_regression,
        model_name="Logistic Regression",
        run_name="Logistic_Regression",
        model_file="models/titanic_logistic_regression.pkl",
        parameters={
            "max_iter": 200,
            "random_state": 42,
            "test_size": 0.2,
        },
    )

    print(
        "\nAll model experiments completed."
    )


if __name__ == "__main__":
    main()