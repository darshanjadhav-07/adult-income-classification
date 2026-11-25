
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Create Graph Output Directory
# --------------------------------------------------

os.makedirs("outputs/graphs", exist_ok=True)

# --------------------------------------------------
# Load Cleaned Dataset
# --------------------------------------------------

df = pd.read_csv("data/adult_cleaned.csv")

print("Dataset shape:", df.shape)

# --------------------------------------------------
# Target Variable Distribution
# --------------------------------------------------

print("\nIncome distribution:")
print(df["income"].value_counts())

print("\nIncome distribution percentage:")
print(df["income"].value_counts(normalize=True) * 100)

# Save income distribution graph
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="income")

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/graphs/income_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# --------------------------------------------------
# Numerical Feature Analysis
# --------------------------------------------------

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumerical columns:")
print(numerical_columns)

print("\nNumerical feature statistics:")
print(df[numerical_columns].describe())

# --------------------------------------------------
# Categorical Feature Analysis
# --------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print("\nCategorical columns:")
print(categorical_columns)

for column in categorical_columns:

    print(f"\n--- {column} ---")

    print("Category counts:")
    print(df[column].value_counts())

    print("\nCategory percentages:")
    print((df[column].value_counts(normalize=True) * 100).round(2))

# --------------------------------------------------
# Categorical Feature Distributions
# --------------------------------------------------

categorical_cols = df.select_dtypes(
    include="object"
).columns.tolist()

# Remove target column
categorical_cols.remove("income")

for col in categorical_cols:

    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        f"outputs/graphs/{col}_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# --------------------------------------------------
# Workclass Distribution by Income
# --------------------------------------------------

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="workclass",
    hue="income"
)

plt.title("Workclass Distribution by Income")
plt.xlabel("Workclass")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/workclass_by_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# Categorical Features vs Income
# --------------------------------------------------

categorical_cols = df.select_dtypes(
    include="object"
).columns.tolist()

# Remove target column
categorical_cols.remove("income")

# Workclass was already plotted
categorical_cols.remove("workclass")

for col in categorical_cols:

    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=df,
        x=col,
        hue="income"
    )

    plt.title(f"{col} Distribution by Income")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        f"outputs/graphs/{col}_by_income.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# --------------------------------------------------
# Numerical Features vs Income
# --------------------------------------------------

numerical_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

for col in numerical_cols:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="income",
        y=col
    )

    plt.title(f"{col} vs Income")
    plt.xlabel("Income")
    plt.ylabel(col)
    plt.tight_layout()

    plt.savefig(
        f"outputs/graphs/{col}_vs_income.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------

numerical_data = df.select_dtypes(
    include=["int64", "float64"]
)

correlation_matrix = numerical_data.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nAll EDA graphs saved successfully in outputs/graphs/")
