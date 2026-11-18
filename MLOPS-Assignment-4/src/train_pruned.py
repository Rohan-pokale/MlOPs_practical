import os
import time
import math
import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from preprocess import load_and_preprocess_data
from mlflow_utils import setup_mlflow, log_common_metrics

# Configuration

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 8
LEARNING_RATE = 0.001
TARGET_SPARSITY = 0.50

MODEL_PATH = "models/pruned_mlp.h5"

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)

# Load data

X_train, X_test, y_train, y_test, _ = (
    load_and_preprocess_data()
)


# MLflow

setup_mlflow()

mlflow.start_run(run_name="Pruned MLP")

try:

    mlflow.log_params({
        "model_type": "MLP",
        "optimization": "Pruning",
        "hidden_layers": "128-64-32-16",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "target_sparsity": TARGET_SPARSITY,
        "random_seed": SEED
    })

       
    # Build model

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Apply pruning 
    
    steps_per_epoch = math.ceil(
        len(X_train) * 0.8 / BATCH_SIZE
    )

    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=TARGET_SPARSITY,
        begin_step=0,
        end_step=steps_per_epoch * 10
    )

    model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule
    )


    # Compile and train

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=False
            )
        ],
        verbose=1
    )

       
    # Strip pruning wrappers
    
    model = tfmot.sparsity.keras.strip_pruning(model)

       
    # Calculate sparsity
       
    total_weights = 0
    zero_weights = 0

    for weight in model.weights:
        values = weight.numpy()
        total_weights += values.size
        zero_weights += np.count_nonzero(values == 0)

    nonzero_weights = total_weights - zero_weights
    sparsity = zero_weights / total_weights

  
    # Evaluate 

    start = time.perf_counter()

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    inference_time = (
        time.perf_counter() - start
    )

    y_pred = (
        probabilities >= 0.5
    ).astype(int).flatten()

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

    avg_inference_ms = (
        inference_time / len(X_test)
    ) * 1000


       
    # Save model and calculate size  

    model.save(MODEL_PATH)

    model_size_kb = (
        os.path.getsize(MODEL_PATH) / 1024
    )

       
    # MLflow metrics   

    log_common_metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        parameter_count=total_weights,
        model_size_kb=model_size_kb,
        inference_time_ms=avg_inference_ms
    )

    mlflow.log_metrics({
        "sparsity": sparsity,
        "zero_parameters": zero_weights,
        "nonzero_parameters": nonzero_weights,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": max(
            history.history["val_accuracy"]
        ),
        "best_validation_loss": min(
            history.history["val_loss"]
        )
    })

    mlflow.log_artifact(MODEL_PATH)
       
    # Results    

    print("\n========== PRUNED MLP ==========")
    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1 Score            : {f1:.4f}")
    print(f"Total Parameters    : {total_weights:,}")
    print(f"Non-zero Parameters : {nonzero_weights:,}")
    print(f"Zero Parameters     : {zero_weights:,}")
    print(f"Sparsity            : {sparsity:.2%}")
    print(f"Model Size          : {model_size_kb:.2f} KB")
    print(f"Avg Inference       : {avg_inference_ms:.4f} ms/sample")
    print("================================")
    print("MLflow run completed successfully.")


finally:
    mlflow.end_run()