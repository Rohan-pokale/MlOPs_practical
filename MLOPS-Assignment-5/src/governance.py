
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

metrics = pd.read_csv(RESULTS / "model_metrics.csv")
config = json.loads(
    (RESULTS / "experiment_config.json").read_text()
)

report = {
    "project": "Titanic Model A/B Experiment",
    "dataset": config["dataset"],
    "purpose": "Educational comparison of survival classifiers",
    "models": config["models"],
    "random_state": config["random_state"],
    "test_size": config["test_size"],
    "evaluation_metrics": metrics.to_dict(orient="records"),
    "governance_checks": {
        "reproducible_split_documented": True,
        "model_artifacts_saved": True,
        "group_fairness_evaluated": (
            (RESULTS / "fairness_LogisticRegression.csv").exists()
            and (RESULTS / "fairness_RandomForest.csv").exists()
        ),
        "explainability_artifact_saved": (
            RESULTS / "feature_importance.csv"
        ).exists(),
        "limitations_documented": True
    },
    "limitations": [
        "Historical Titanic data is not representative of modern populations.",
        "Observed group differences do not establish causation.",
        "Results do not establish regulatory compliance.",
        "Test-set results may vary on other datasets."
    ],
    "privacy_and_compliance": (
        "Use only for educational analysis; do not use for real-world "
        "eligibility or consequential decisions."
    )
}

(RESULTS / "governance_report.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
print("\nGovernance report saved.")
 