from preprocessing import load_data, prepare_data, create_preprocessor


DATA_PATH = "data/Titanic-Dataset.csv"


# Load dataset
df = load_data(DATA_PATH)

print("Original shape:", df.shape)


# Prepare features and target
X, y = prepare_data(df)

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nMissing values before preprocessing:")
print(X.isnull().sum())


# Create preprocessing pipeline
preprocessor = create_preprocessor()


# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

print("\nProcessed shape:", X_processed.shape)

print("\nPreprocessing completed successfully.")