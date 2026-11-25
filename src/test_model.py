import pandas as pd
import joblib

# Load saved model
model = joblib.load("models/random_forest_model.pkl")

# Load cleaned dataset for a sample input
df = pd.read_csv("data/adult_cleaned.csv")

# Select one sample record (excluding target)
sample = df.drop("income", axis=1).iloc[[0]]

# Predict income
prediction = model.predict(sample)

print("Sample input:")
print(sample)

print("\nPredicted Income:", prediction[0])