import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import load_data, prepare_data


DATA_PATH = "data/Titanic-Dataset.csv"


# --------------------------------------------------
# PSI Calculation
# --------------------------------------------------

def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI)
    between a reference distribution and a current
    distribution.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    # Create bins from reference data
    breakpoints = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1)
    )

    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)

    # If there are not enough unique values,
    # PSI cannot be calculated using these bins.
    if len(breakpoints) < 3:
        return 0.0

    # Make the first and last boundaries open-ended
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_binned = pd.cut(
        reference,
        bins=breakpoints,
        include_lowest=True
    )

    current_binned = pd.cut(
        current,
        bins=breakpoints,
        include_lowest=True
    )

    reference_distribution = (
        reference_binned
        .value_counts(normalize=True)
        .sort_index()
    )

    current_distribution = (
        current_binned
        .value_counts(normalize=True)
        .sort_index()
    )

    # Make sure both distributions contain
    # exactly the same bins
    all_bins = reference_distribution.index.union(
        current_distribution.index
    )

    reference_distribution = (
        reference_distribution
        .reindex(all_bins, fill_value=0)
    )

    current_distribution = (
        current_distribution
        .reindex(all_bins, fill_value=0)
    )

    # Avoid log(0)
    epsilon = 0.0001

    reference_distribution = np.clip(
        reference_distribution,
        epsilon,
        None
    )

    current_distribution = np.clip(
        current_distribution,
        epsilon,
        None
    )

    psi = np.sum(
        (current_distribution - reference_distribution)
        * np.log(
            current_distribution / reference_distribution
        )
    )

    return float(psi)


# --------------------------------------------------
# Categorical PSI
# --------------------------------------------------

def calculate_categorical_psi(reference, current):
    """
    Calculate PSI for categorical data.
    """

    reference = pd.Series(reference).fillna("Missing")
    current = pd.Series(current).fillna("Missing")

    categories = set(reference.unique()).union(
        set(current.unique())
    )

    reference_distribution = (
        reference
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    current_distribution = (
        current
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    epsilon = 0.0001

    reference_distribution = np.clip(
        reference_distribution,
        epsilon,
        None
    )

    current_distribution = np.clip(
        current_distribution,
        epsilon,
        None
    )

    psi = np.sum(
        (current_distribution - reference_distribution)
        * np.log(
            current_distribution / reference_distribution
        )
    )

    return float(psi)


# --------------------------------------------------
# Drift Detection
# --------------------------------------------------

def detect_drift(reference_data, current_data):

    numerical_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]

    categorical_features = [
        "Sex",
        "Embarked",
    ]

    results = {}

    # Numerical features
    for feature in numerical_features:

        results[feature] = calculate_psi(
            reference_data[feature],
            current_data[feature]
        )

    # Categorical features
    for feature in categorical_features:

        results[feature] = calculate_categorical_psi(
            reference_data[feature],
            current_data[feature]
        )

    # Overall drift score
    overall_psi = np.mean(
        list(results.values())
    )

    return results, float(overall_psi)


# --------------------------------------------------
# Main Demonstration
# --------------------------------------------------

if __name__ == "__main__":

    # Load Titanic dataset
    df = load_data(DATA_PATH)

    # Prepare features
    X, y = prepare_data(df)

    # Same split used during model training
    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # Normal current data
    # --------------------------------------------------

    normal_results, normal_psi = detect_drift(
        X_train,
        X_test
    )

    print("\nNormal Data Drift")
    print("=================")

    for feature, psi in normal_results.items():

        print(
            f"{feature:10s}: {psi:.4f}"
        )

    print(
        f"\nOverall PSI: {normal_psi:.4f}"
    )


    # --------------------------------------------------
    # Simulated drifted data
    # --------------------------------------------------

    drifted_data = X_test.copy()

    # Simulate a change in Age distribution
    drifted_data["Age"] = (
        drifted_data["Age"] + 25
    ).clip(0, 100)

    # Simulate a change in Fare distribution
    drifted_data["Fare"] = (
        drifted_data["Fare"] * 3
    )

    drifted_results, drifted_psi = detect_drift(
        X_train,
        drifted_data
    )

    print("\nDrifted Data")
    print("============")

    for feature, psi in drifted_results.items():

        print(
            f"{feature:10s}: {psi:.4f}"
        )

    print(
        f"\nOverall PSI: {drifted_psi:.4f}"
    )


    # --------------------------------------------------
    # Drift Interpretation
    # --------------------------------------------------

    print("\nDrift Interpretation")
    print("====================")

    if drifted_psi < 0.10:

        print("No significant drift detected.")

    elif drifted_psi < 0.20:

        print("Moderate drift detected.")

    else:

        print("Significant drift detected.")