from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML model serving API for Titanic survival prediction",
    version="1.0.0",
)

# Load trained model
model = joblib.load("models/titanic_model.pkl")


class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(passenger: Passenger):

    data = pd.DataFrame([
        {
            "Pclass": passenger.Pclass,
            "Sex": passenger.Sex,
            "Age": passenger.Age,
            "SibSp": passenger.SibSp,
            "Parch": passenger.Parch,
            "Fare": passenger.Fare,
            "Embarked": passenger.Embarked,
        }
    ])

    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction),
        "result": (
            "Passenger Survived"
            if prediction == 1
            else "Passenger Not Survived"
        ),
    }