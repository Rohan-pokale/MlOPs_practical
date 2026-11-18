import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/Titanic-Dataset.csv"


def load_and_preprocess_data():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # Remove unnecessary columns
    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    df = df.drop(columns=columns_to_drop)

    # Handle missing values
    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )

    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({
        "male": 0,
        "female": 1
    })

    df["Embarked"] = df["Embarked"].map({
        "S": 0,
        "C": 1,
        "Q": 2
    })

    # Features
    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    X = df[feature_columns]
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardization
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert targets to NumPy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )


if __name__ == "__main__":
    load_and_preprocess_data()