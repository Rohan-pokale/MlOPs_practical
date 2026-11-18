# MLOps Assignment 5 – Model Evaluation and Governance

## Overview

This project evaluates two machine learning classification models using the Titanic dataset.

It implements A/B testing, statistical evaluation, model explainability, bias detection, and model governance to assess model performance beyond accuracy.

## Project Structure

```text
MLOPS_5/
│
├── data/
│   └── Titanic-Dataset.csv
│
├── src/
│   ├── train_models.py
│   ├── ab_test.py
│   ├── explainability.py
│   ├── fairness.py
│   └── governance.py
│
├── models/
│   └── Trained model artifacts
│
├── results/
│   ├── Model evaluation results
│   ├── Feature importance outputs
│   └── governance_report.json
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## Main Components

- **Model Training:** Trains Logistic Regression and Random Forest models.
- **A/B Testing:** Compares model predictions using McNemar's exact test and bootstrap confidence intervals.
- **Explainability:** Uses permutation feature importance to identify influential features.
- **Fairness Assessment:** Uses Fairlearn to evaluate model performance across female and male groups.
- **Model Governance:** Generates a JSON report documenting reproducibility, evaluation checks, and limitations.

## Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- Fairlearn
- Matplotlib
- SciPy
- Rich
- uv

## How to Run

**1. Clone or open the project directory.**

**2. Install dependencies:**

```bash
uv sync
```

**3. Run model training:**

```bash
uv run python src/train_models.py
```

**4. Run the evaluation components:**

```bash
uv run python src/ab_test.py
uv run python src/explainability.py
uv run python src/fairness.py
uv run python src/governance.py
```

Generated models are stored in `models/`, while evaluation results and reports are stored in `results/`.

## Dataset

The project uses the Titanic dataset containing passenger information to predict survival. The target variable is `Survived`.