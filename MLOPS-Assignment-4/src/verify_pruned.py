import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import load_and_preprocess_data


MODEL_PATH = "models/pruned_mlp.h5"


# Load data
X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

# Load saved pruned model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# --------------------------------------------------
# Model information
# --------------------------------------------------

total_weights = 0
zero_weights = 0

for weight in model.weights:
    values = weight.numpy()

    total_weights += values.size
    zero_weights += np.count_nonzero(values == 0)

nonzero_weights = total_weights - zero_weights
sparsity = zero_weights / total_weights


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

probabilities = model.predict(X_test, verbose=0)

y_pred = (probabilities >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)


# --------------------------------------------------
# File information
# --------------------------------------------------

model_size_kb = os.path.getsize(MODEL_PATH) / 1024


# --------------------------------------------------
# Print verification
# --------------------------------------------------

print("\n========== PRUNED MODEL VERIFICATION ==========")

print(f"Model path          : {MODEL_PATH}")
print(f"Model size          : {model_size_kb:.2f} KB")

print("\nModel structure:")
model.summary()

print("\nParameter verification:")
print(f"Total parameters    : {total_weights:,}")
print(f"Non-zero parameters : {nonzero_weights:,}")
print(f"Zero parameters     : {zero_weights:,}")
print(f"Sparsity            : {sparsity:.2%}")

print("\nPerformance verification:")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")

print("===============================================")