import mlflow


EXPERIMENT_NAME = "Titanic_Model_Optimization"


def setup_mlflow():

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def log_common_metrics(
    accuracy,
    precision,
    recall,
    f1,
    parameter_count,
    model_size_kb,
    inference_time_ms
):

    mlflow.log_metrics({

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "parameter_count": parameter_count,

        "model_size_kb": model_size_kb,

        "inference_time_ms": inference_time_ms
    })