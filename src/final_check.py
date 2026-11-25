import pandas as pd
import joblib
import os

# --------------------------------------------------
# Check Dataset
# --------------------------------------------------

df = pd.read_csv("data/adult_cleaned.csv")

print("Dataset validation")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

# --------------------------------------------------
# Check Model Comparison File
# --------------------------------------------------

comparison = pd.read_csv("outputs/model_comparison.csv")

print("\nModel comparison validation")
print(comparison)
print("Number of models:", len(comparison))

# --------------------------------------------------
# Check Feature Importance File
# --------------------------------------------------

importance = pd.read_csv("outputs/feature_importance.csv")

print("\nFeature importance validation")
print("Number of features:", len(importance))
print("Top 5 features:")
print(importance.head())

# --------------------------------------------------
# Check Saved Model
# --------------------------------------------------

model = joblib.load("models/random_forest_model.pkl")

print("\nSaved model validation")
print("Model loaded successfully.")

# --------------------------------------------------
# Check Project Files
# --------------------------------------------------

files = [
    "data/adult.data",
    "data/adult_cleaned.csv",
    "models/random_forest_model.pkl",
    "outputs/model_comparison.csv",
    "outputs/feature_importance.csv",
    "README.md"
]

print("\nFile existence check")

for file in files:
    print(file, ":", os.path.exists(file))

print("\nFINAL VALIDATION COMPLETED")