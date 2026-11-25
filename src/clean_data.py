
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "adult.data"
OUTPUT_PATH = PROJECT_ROOT / "data" / "adult_cleaned.csv"

# --------------------------------------------------
# 2. Define column names
# --------------------------------------------------

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

# --------------------------------------------------
# 3. Load raw dataset
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    names=columns,
    skipinitialspace=True
)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 4. Replace '?' with proper missing values
# --------------------------------------------------

df.replace("?", pd.NA, inplace=True)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# 5. Remove duplicate records
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate records found:", duplicate_count)

df.drop_duplicates(inplace=True)

print("Shape after removing duplicates:", df.shape)

# --------------------------------------------------
# 6. Handle missing categorical values
# --------------------------------------------------

categorical_columns = [
    "workclass",
    "occupation",
    "native-country"
]

df[categorical_columns] = df[categorical_columns].fillna("Unknown")

# --------------------------------------------------
# 7. Clean target variable whitespace
# --------------------------------------------------

df["income"] = df["income"].str.strip()

# --------------------------------------------------
# 8. Validate cleaned dataset
# --------------------------------------------------

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate records after cleaning:")
print(df.duplicated().sum())

print("\nFinal dataset shape:")
print(df.shape)

print("\nIncome distribution:")
print(df["income"].value_counts())

# --------------------------------------------------
# 9. Save cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved to:")
print(OUTPUT_PATH)
