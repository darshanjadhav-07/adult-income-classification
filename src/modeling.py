import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Create output directories
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/adult_cleaned.csv")

X = df.drop("income", axis=1)
y = df["income"]

# --------------------------------------------------
# 3. Identify numerical and categorical features
# --------------------------------------------------

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Convert positive class into binary values for ROC-AUC
y_test_binary = (y_test == ">50K").astype(int)

# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# --------------------------------------------------
# 6. Helper function for Naive Bayes
# --------------------------------------------------

def convert_to_dense(X):
    if issparse(X):
        return X.toarray()
    return X

# --------------------------------------------------
# 7. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),

    "KNN": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": Pipeline([
        ("preprocessor", preprocessor),
        ("dense", FunctionTransformer(convert_to_dense, validate=False)),
        ("classifier", GaussianNB())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# --------------------------------------------------
# 8. Train and evaluate models
# --------------------------------------------------

results = []
predictions = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # Get probability for positive class
    y_prob = model.predict_proba(X_test)

    positive_class_index = list(model.classes_).index(">50K")
    positive_prob = y_prob[:, positive_class_index]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=">50K")
    recall = recall_score(y_test, y_pred, pos_label=">50K")
    f1 = f1_score(y_test, y_pred, pos_label=">50K")
    auc = roc_auc_score(y_test_binary, positive_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc,
        "MCC": mcc
    })

    # Display metrics
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"MCC       : {mcc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 9. Save model comparison
# --------------------------------------------------

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="F1-score",
    ascending=False
)

comparison_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison_df.to_string(index=False))

# --------------------------------------------------
# 10. Generate metric comparison graphs
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "MCC"
]

for metric in metrics:

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df[metric]
    )

    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Models")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        f"outputs/graphs/{metric.lower().replace('-', '_')}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# --------------------------------------------------
# 11. Random Forest Feature Importance
# --------------------------------------------------

rf_model = models["Random Forest"]

rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["classifier"]

feature_names = rf_preprocessor.get_feature_names_out()
importances = rf_classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

feature_importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# Plot top 15 features
top_features = feature_importance_df.head(15)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 12. Save Random Forest model
# --------------------------------------------------

joblib.dump(
    rf_model,
    "models/random_forest_model.pkl"
)

print("\nRandom Forest model saved successfully!")

print("\nTop 15 Important Features:")
print(top_features.to_string(index=False))

print("\nModel evaluation completed successfully!")