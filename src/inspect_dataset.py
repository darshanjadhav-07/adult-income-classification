import pandas as pd

# Define column names for the Adult Census Income dataset
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

# Load the dataset
data_path = "data/adult.data"

df = pd.read_csv(
    data_path,
    names=column_names,
    skipinitialspace=True
)

# Display basic dataset information
print("Dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()