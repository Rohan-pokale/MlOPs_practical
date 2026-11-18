from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)
from fastapi.responses import Response

import joblib
import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("src")

from model_monitoring import evaluate_model
from drift_detection import detect_drift
from anomaly_detection import (
    prepare_numerical_data,
    train_anomaly_detector,
    detect_anomalies,
)
from preprocessing import load_data, prepare_data


app = FastAPI(title="Titanic MLOps Monitoring API")


DATA_PATH = "data/Titanic-Dataset.csv"
MODEL_PATH = "models/decision_tree.pkl"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Prometheus Operational Metrics
# --------------------------------------------------

prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency"
)

predictions_total = Counter(
    "predictions_total",
    "Total predictions made",
    ["prediction"]
)


# --------------------------------------------------
# Prometheus Model Performance Metrics
# --------------------------------------------------

model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy"
)

model_precision = Gauge(
    "model_precision",
    "Current model precision"
)

model_recall = Gauge(
    "model_recall",
    "Current model recall"
)

model_f1_score = Gauge(
    "model_f1_score",
    "Current model F1 score"
)


# --------------------------------------------------
# Prometheus Data Drift Metrics
# --------------------------------------------------

data_drift_score = Gauge(
    "data_drift_score",
    "Overall Population Stability Index for data drift"
)

data_drift_detected = Gauge(
    "data_drift_detected",
    "Whether significant data drift was detected"
)


# --------------------------------------------------
# Prometheus Anomaly Metrics
# --------------------------------------------------

anomalies_detected = Gauge(
    "anomalies_detected",
    "Number of anomalous records detected"
)

anomaly_detection_rate = Gauge(
    "anomaly_detection_rate",
    "Percentage of records identified as anomalies"
)


# --------------------------------------------------
# Model Performance Initialization
# --------------------------------------------------

performance = evaluate_model()

model_accuracy.set(performance["accuracy"])
model_precision.set(performance["precision"])
model_recall.set(performance["recall"])
model_f1_score.set(performance["f1_score"])


# --------------------------------------------------
# Request Models
# --------------------------------------------------

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


class DriftRequest(BaseModel):
    scenario: str


class AnomalyRequest(BaseModel):
    scenario: str


# --------------------------------------------------
# Basic Endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Titanic MLOps Monitoring API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(passenger: Passenger):

    start_time = time.time()

    prediction_requests.inc()

    try:

        input_df = pd.DataFrame(
            [[
                passenger.Pclass,
                passenger.Sex,
                passenger.Age,
                passenger.SibSp,
                passenger.Parch,
                passenger.Fare,
                passenger.Embarked
            ]],
            columns=[
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked"
            ]
        )

        prediction = int(model.predict(input_df)[0])

        predictions_total.labels(
            prediction=str(prediction)
        ).inc()

        return {
            "prediction": prediction,
            "meaning": (
                "Survived"
                if prediction == 1
                else "Did not survive"
            )
        }

    except Exception:

        prediction_errors.inc()

        raise

    finally:

        prediction_latency.observe(
            time.time() - start_time
        )


# --------------------------------------------------
# Data Drift Monitoring
# --------------------------------------------------

@app.post("/monitor/drift")
def monitor_drift(request: DriftRequest):

    df = load_data(DATA_PATH)

    X, y = prepare_data(df)

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    if request.scenario.lower() == "normal":

        current_data = X_test.copy()

    elif request.scenario.lower() == "drifted":

        current_data = X_test.copy()

        current_data["Age"] = (
            current_data["Age"] + 25
        ).clip(0, 100)

        current_data["Fare"] = (
            current_data["Fare"] * 3
        )

    else:

        return {
            "error": "scenario must be 'normal' or 'drifted'"
        }

    feature_results, overall_psi = detect_drift(
        X_train,
        current_data
    )

    drift_detected = (
        1 if overall_psi >= 0.20 else 0
    )

    data_drift_score.set(overall_psi)

    data_drift_detected.set(drift_detected)

    return {
        "scenario": request.scenario.lower(),
        "overall_psi": round(overall_psi, 4),
        "drift_detected": bool(drift_detected),
        "feature_psi": {
            feature: round(score, 4)
            for feature, score in feature_results.items()
        }
    }


# --------------------------------------------------
# Anomaly Monitoring
# --------------------------------------------------

@app.post("/monitor/anomaly")
def monitor_anomaly(request: AnomalyRequest):

    df = load_data(DATA_PATH)

    X, y = prepare_data(df)

    reference_data = prepare_numerical_data(X)

    detector = train_anomaly_detector(
        reference_data
    )

    if request.scenario.lower() == "normal":

        current_data = reference_data.sample(
            n=100,
            random_state=42
        ).copy()

    elif request.scenario.lower() == "anomalous":

        current_data = reference_data.sample(
            n=100,
            random_state=42
        ).copy()

        # Inject anomalies into 10% of the records
        anomaly_indices = current_data.index[:10]

        current_data.loc[
            anomaly_indices,
            "Age"
        ] = (
            current_data.loc[
                anomaly_indices,
                "Age"
            ] + 60
        ).clip(0, 120)

        current_data.loc[
            anomaly_indices,
            "Fare"
        ] = (
            current_data.loc[
                anomaly_indices,
                "Fare"
            ] * 10
        )

    else:

        return {
            "error": (
                "scenario must be "
                "'normal' or 'anomalous'"
            )
        }

    anomaly_count, predictions = detect_anomalies(
        detector,
        current_data
    )

    anomaly_rate = (
        anomaly_count / len(current_data)
    )

    anomalies_detected.set(
        anomaly_count
    )

    anomaly_detection_rate.set(
        anomaly_rate
    )

    return {
        "scenario": request.scenario.lower(),
        "records_checked": len(current_data),
        "anomalies_detected": anomaly_count,
        "anomaly_rate": round(
            anomaly_rate,
            4
        )
    }   


# --------------------------------------------------
# Prometheus Metrics Endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )