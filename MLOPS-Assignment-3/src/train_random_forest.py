import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import load_data, prepare_data, create_preprocessor


DATA_PATH = "data/Titanic-Dataset.csv"
MODEL_PATH = "models/random_forest.pkl"


# 1. Load dataset
df = load_data(DATA_PATH)

# 2. Prepare features and target
X, y = prepare_data(df)

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Create preprocessing + Random Forest pipeline
model_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# 5. Train model
model_pipeline.fit(X_train, y_train)

print("\nModel training completed.")


# 6. Make predictions
y_pred = model_pipeline.predict(X_test)


# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nRandom Forest Results")
print("---------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# 8. Save complete pipeline
joblib.dump(model_pipeline, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")