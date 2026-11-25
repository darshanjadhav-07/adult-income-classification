import pandas as pd

column_names = [
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

df = pd.read_csv(
    "data/adult.data",
    names=column_names,
    skipinitialspace=True
)

# 1. Check missing values represented by '?'
print("Missing values represented by '?':")
missing_values = (df == "?").sum()
print(missing_values)

# 2. Check duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())

# 3. Check target distribution
print("\nIncome distribution:")
print(df["income"].value_counts())

# 4. Check target distribution in percentage
print("\nIncome distribution (%):")
print(df["income"].value_counts(normalize=True) * 100)

# 5. Numerical feature summary
print("\nNumerical feature summary:")
print(df.describe())

# 6. Categorical feature summary
print("\nCategorical feature summary:")
print(df.describe(include="object"))

# Missing-value percentages
print("\nMissing-value percentages:")

missing_percentage = (df == "?").mean() * 100

print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

# Display some duplicate records
print("\nSample duplicate records:")

duplicate_rows = df[df.duplicated()]

print(duplicate_rows.head(10))

# Verify the unique target labels
print("\nUnique income labels:")
print(df["income"].unique())