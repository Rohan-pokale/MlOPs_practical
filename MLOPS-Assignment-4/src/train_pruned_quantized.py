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
from mlflow_utils import setup_mlflow, log_common_metrics


MODEL_PATH = "models/pruned_mlp.h5"
FP32_PATH = "models/pruned_fp32.tflite"
QUANTIZED_PATH = "models/pruned_quantized_mlp.tflite"


X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

setup_mlflow()
mlflow.start_run(run_name="Pruned + Quantized MLP")

try:
    mlflow.log_params({
        "optimization": "Pruning + Quantization",
        "quantization_type": "Dynamic Range",
        "source_model": "pruned_mlp.h5"
    })

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_model = converter.convert()

    with open(FP32_PATH, "wb") as f:
        f.write(fp32_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_model = converter.convert()

    with open(QUANTIZED_PATH, "wb") as f:
        f.write(quantized_model)

    interpreter = tf.lite.Interpreter(
        model_path=QUANTIZED_PATH
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    predictions = []

    start = time.perf_counter()

    for sample in X_test.astype(np.float32):
        sample = np.expand_dims(sample, axis=0)

        interpreter.set_tensor(
            input_index,
            sample
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )

        predictions.append(output[0][0])

    inference_time = (
        time.perf_counter() - start
    )

    y_pred = (
        np.array(predictions) >= 0.5
    ).astype(int)

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

    avg_inference_ms = (
        inference_time / len(X_test)
    ) * 1000

    fp32_size_kb = (
        os.path.getsize(FP32_PATH) / 1024
    )

    quantized_size_kb = (
        os.path.getsize(QUANTIZED_PATH) / 1024
    )

    size_reduction = (
        (fp32_size_kb - quantized_size_kb)
        / fp32_size_kb
    ) * 100

    total_weights = 0
    zero_weights = 0

    for weight in model.weights:
        values = weight.numpy()

        total_weights += values.size
        zero_weights += np.count_nonzero(
            values == 0
        )

    nonzero_weights = (
        total_weights - zero_weights
    )

    sparsity = (
        zero_weights / total_weights
    )

    log_common_metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        parameter_count=total_weights,
        model_size_kb=quantized_size_kb,
        inference_time_ms=avg_inference_ms
    )

    mlflow.log_metrics({
        "sparsity": sparsity,
        "zero_parameters": zero_weights,
        "nonzero_parameters": nonzero_weights,
        "fp32_tflite_size_kb": fp32_size_kb,
        "quantized_tflite_size_kb": quantized_size_kb,
        "size_reduction_percent": size_reduction
    })

    mlflow.log_artifact(FP32_PATH)
    mlflow.log_artifact(QUANTIZED_PATH)

    print("\n======= PRUNED + QUANTIZED MLP =======")
    print(f"Accuracy              : {accuracy:.4f}")
    print(f"Precision             : {precision:.4f}")
    print(f"Recall                : {recall:.4f}")
    print(f"F1 Score              : {f1:.4f}")
    print(f"Parameters            : {total_weights:,}")
    print(f"Non-zero Parameters   : {nonzero_weights:,}")
    print(f"Zero Parameters       : {zero_weights:,}")
    print(f"Sparsity              : {sparsity:.2%}")
    print(f"FP32 TFLite Size      : {fp32_size_kb:.2f} KB")
    print(f"Quantized Model Size  : {quantized_size_kb:.2f} KB")
    print(f"Size Reduction        : {size_reduction:.2f}%")
    print(f"Avg Inference         : {avg_inference_ms:.4f} ms/sample")
    print("=======================================")
    print("MLflow run completed successfully.")

finally:
    mlflow.end_run()