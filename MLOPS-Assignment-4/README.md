# MLOps Assignment 4 – Model Performance Optimization

## Overview
This project explores model optimization techniques using a Titanic survival prediction dataset and an MLP model.

The following approaches are implemented:
- **Baseline:** Original MLP model.
- **Pruning:** Introduces sparsity by setting selected weights to zero.
- **Quantization:** Reduces model size using TensorFlow Lite dynamic-range quantization.
- **Pruning + Quantization:** Combines both optimization techniques.

## Technologies
- Python
- TensorFlow / Keras
- TensorFlow Model Optimization
- TensorFlow Lite
- Scikit-learn
- MLflow
- uv

## Project Structure
```text
MLOps_Assignment_4/
├── data/       # Titanic dataset
├── models/     # Saved models
├── src/        # Source code
├── pyproject.toml
└── README.md
```

## How to Run

1. Install dependencies:
```bash
uv sync
```

2. Run the scripts from the project root:
```bash
uv run python src/train_baseline.py
uv run python src/train_pruned.py
uv run python src/train_quantized.py
uv run python src/train_pruned_quantized.py
```

3. Launch MLflow:
```bash
uv run mlflow ui
```

Open `http://127.0.0.1:5000` to view experiment results.

## Results
- Pruning achieved **44.70% sparsity**.
- Quantization reduced TFLite model size by approximately **58%**.
- Combined optimization achieved **80.45% accuracy** and **72.00% F1 score**.