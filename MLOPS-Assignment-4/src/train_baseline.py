import os
import time

import mlflow
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import load_and_preprocess_data

from mlflow_utils import (
    setup_mlflow,
    log_common_metrics
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 8

MODEL_PATH = "models/baseline_mlp.keras"


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

X_train, X_test, y_train, y_test, scaler = (
    load_and_preprocess_data()
)


# ============================================================
# SETUP MLFLOW
# ============================================================

setup_mlflow()

mlflow.start_run(
    run_name="Baseline MLP"
)


try:

    # ========================================================
    # LOG PARAMETERS
    # ========================================================

    mlflow.log_params({

        "model_type": "MLP",

        "optimization": "None",

        "hidden_layers": "128-64-32-16",

        "learning_rate": LEARNING_RATE,

        "batch_size": BATCH_SIZE,

        "max_epochs": MAX_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "validation_split": 0.2,

        "random_seed": SEED,

        "optimizer": "Adam",

        "loss_function":
            "binary_crossentropy"
    })


    # ========================================================
    # BUILD BASELINE MODEL
    # ========================================================

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(7,)
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            name="dense_128"
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu",
            name="dense_64"
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu",
            name="dense_32"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu",
            name="dense_16"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="output"
        )
    ])


    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=["accuracy"]
    )


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    print(
        "\n========== BASELINE MODEL ==========\n"
    )

    model.summary()


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=EARLY_STOPPING_PATIENCE,

            restore_best_weights=True
        )
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n========== TRAINING ==========\n"
    )

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.2,

        epochs=MAX_EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[early_stopping],

        verbose=1
    )


    # ========================================================
    # TRAINING METRICS
    # ========================================================

    epochs_completed = len(
        history.history["loss"]
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    mlflow.log_metrics({

        "epochs_completed":
            epochs_completed,

        "best_validation_accuracy":
            best_val_accuracy,

        "best_validation_loss":
            best_val_loss
    })


    # ========================================================
    # EVALUATION
    # ========================================================

    print(
        "\n========== EVALUATION ==========\n"
    )

    start_time = time.perf_counter()

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    end_time = time.perf_counter()


    # Convert probabilities to classes
    y_pred = (
        probabilities >= 0.5
    ).astype(int).flatten()


    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # ========================================================
    # INFERENCE TIME
    # ========================================================

    total_inference_time = (
        end_time - start_time
    )

    average_inference_time = (
        total_inference_time
        / len(X_test)
    ) * 1000


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    parameter_count = model.count_params()


    # Save model
    model.save(MODEL_PATH)


    # Calculate model size
    model_size_bytes = os.path.getsize(
        MODEL_PATH
    )

    model_size_kb = (
        model_size_bytes / 1024
    )


    # ========================================================
    # LOG METRICS TO MLFLOW
    # ========================================================

    log_common_metrics(

        accuracy=accuracy,

        precision=precision,

        recall=recall,

        f1=f1,

        parameter_count=parameter_count,

        model_size_kb=model_size_kb,

        inference_time_ms=average_inference_time
    )


    # ========================================================
    # LOG MODEL ARTIFACT
    # ========================================================

    mlflow.log_artifact(
        MODEL_PATH
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "       BASELINE MODEL RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Epochs Completed   : "
        f"{epochs_completed}"
    )

    print(
        f"Best Val Accuracy  : "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best Val Loss      : "
        f"{best_val_loss:.4f}"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision          : "
        f"{precision:.4f}"
    )

    print(
        f"Recall             : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{f1:.4f}"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Parameters         : "
        f"{parameter_count:,}"
    )

    print(
        f"Model Size         : "
        f"{model_size_kb:.2f} KB"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Total Inference    : "
        f"{total_inference_time:.6f} sec"
    )

    print(
        f"Avg Inference      : "
        f"{average_inference_time:.6f} ms/sample"
    )

    print(
        "========================================"
    )

    print(
        "\nBaseline model saved to:"
    )

    print(MODEL_PATH)

    print(
        "\nMLflow run completed successfully."
    )


finally:

    mlflow.end_run()