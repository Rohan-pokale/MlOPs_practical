import pandas as pd

DATA_PATH = "data/Titanic-Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Survived"].value_counts())

print("\nNumerical Feature Summary:")
print(df[["Age", "Fare", "Pclass", "SibSp", "Parch"]].describe())

print("\nCategorical Feature Distributions:")

for column in ["Sex", "Embarked"]:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))