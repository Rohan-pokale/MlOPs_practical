# Titanic ML Classification with MLflow

## MLOps Assignment 2: Experiment Tracking and Management

This project demonstrates machine learning experiment tracking and management using MLflow on the Titanic dataset.

## Objective

The objective is to train multiple machine learning models and use MLflow to track parameters, metrics, artifacts, and trained models for experiment comparison, reproducibility, and model registration.

## Dataset

Dataset used:

- Titanic Dataset
- File: `data/Titanic-Dataset.csv`
- Samples: 891
- Features: 11 before preprocessing

## Preprocessing

The following preprocessing steps were performed:

1. Removed unnecessary columns:
   - PassengerId
   - Name
   - Ticket
   - Cabin

2. Missing values:
   - Age → median
   - Embarked → mode

3. Label encoding:
   - Sex
   - Embarked

4. Train-test split:
   - Training data: 80%
   - Testing data: 20%
   - Random state: 42

## Machine Learning Models

Three classification models were evaluated:

1. Random Forest
2. Decision Tree
3. Logistic Regression

## MLflow Experiments

### Experiment 1: Titanic_Classification

The following models were tracked:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.8212 | 0.8088 | 0.7432 | 0.7746 |
| Decision Tree | 0.7989 | 0.8276 | 0.6486 | 0.7273 |
| Logistic Regression | 0.8101 | 0.7857 | 0.7432 | 0.7639 |

Random Forest achieved the best overall accuracy and F1 score among the three baseline models.

### Experiment 2: Titanic_Hyperparameter_Tuning

Random Forest hyperparameters were explored using multiple configurations:

| n_estimators | max_depth | Accuracy | F1 Score |
|---:|---:|---:|---:|
| 50 | 3 | 0.8101 | 0.7536 |
| 100 | 5 | 0.8156 | 0.7556 |
| 150 | 10 | 0.8101 | 0.7536 |
| 200 | None | 0.8101 | 0.7671 |

The original Random Forest baseline remained the best overall model with an accuracy of 0.8212 and F1 score of 0.7746.

## Model Registry

The selected Random Forest model was registered in MLflow as:

`Titanic_Classifier`

Registered version:

`Version 1`

## Reproducibility

The project uses fixed experiment settings including:

- Random state = 42
- Test size = 0.2
- Recorded model parameters
- Recorded evaluation metrics
- MLflow experiment and run names
- Logged confusion matrices
- Logged classification reports
- Logged trained models

The training code is maintained in:

`src/train.py`

## Project Structure

```text
MLOPS-MLFLOW/
│
├── data/
│   └── Titanic-Dataset.csv
│
├── src/
│   └── train.py
│
├── models/
│   ├── titanic_model.pkl
│   ├── titanic_decision_tree.pkl
│   └── titanic_logistic_regression.pkl
│
├── screenshots/
│
├── mlflow.db
├── pyproject.toml
├── uv.lock
└── README.md