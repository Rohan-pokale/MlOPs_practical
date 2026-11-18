import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from preprocessing import load_data, prepare_data

DATA_PATH = "data/Titanic-Dataset.csv"

NUMERICAL_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]


def prepare_numerical_data(df):
    data = df[NUMERICAL_FEATURES].copy()

    # Fill missing numerical values using the median
    data = data.fillna(data.median())

    return data


def train_anomaly_detector(reference_data):
    detector = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    detector.fit(reference_data)

    return detector


def detect_anomalies(detector, data):
    predictions = detector.predict(data)

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly
    anomaly_count = int(np.sum(predictions == -1))

    return anomaly_count, predictions


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X, y = prepare_data(df)

    reference_data = prepare_numerical_data(X)

    detector = train_anomaly_detector(reference_data)

    anomaly_count, predictions = detect_anomalies(
        detector,
        reference_data
    )

    print("\nAnomaly Detection")
    print("=================")
    print(f"Total records : {len(reference_data)}")
    print(f"Anomalies     : {anomaly_count}")
    print(f"Normal        : {len(reference_data) - anomaly_count}") 