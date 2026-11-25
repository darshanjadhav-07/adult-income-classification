# Adult Income Classification

## 1. Project Overview

The Adult Income Classification project aims to predict whether an individual earns more than $50K per year based on demographic and employment-related attributes.

This project applies machine learning classification algorithms to the Adult Income dataset and compares the performance of five different models using multiple evaluation metrics.

The objective is to identify a suitable classification model for predicting income categories based on the available features.

---

## 2. Dataset Description

The dataset used in this project is the Adult Income dataset.

The dataset contains demographic and employment-related information about individuals, including:

- Age
- Workclass
- Final Weight (fnlwgt)
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours Per Week
- Native Country

### Target Variable

The target variable is `income`, which contains two classes:

- `<=50K`
- `>50K`

The classification task is to predict an individual's income category based on the input features.

### Dataset Information

- Dataset file: `adult_cleaned.csv`
- Number of records: 32,537
- Number of features: 14 input features
- Target variable: `income`
- Number of classes: 2

---

## 3. Data Preprocessing

Data preprocessing was performed to prepare the dataset for machine learning model training.

The preprocessing steps included:

- Loading the cleaned dataset.
- Separating the input features and target variable.
- Identifying numerical and categorical features.
- Splitting the dataset into training and testing datasets.
- Applying StandardScaler to numerical features.
- Applying OneHotEncoder to categorical features.

### Train-Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

The split was performed using stratified sampling with a random state of 42 to maintain the distribution of income classes across both datasets.

### Feature Transformation

- StandardScaler was used to scale numerical features.
- OneHotEncoder was used to transform categorical features into numerical representations.
- `handle_unknown="ignore"` was used to handle unseen categorical values during prediction.

The preprocessing steps were integrated into Scikit-learn pipelines to ensure consistent transformation during training and evaluation.

---

## 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the distribution of features and identify patterns associated with income.

The EDA included:

- Income class distribution.
- Distribution of categorical features.
- Distribution of numerical features.
- Feature comparisons with the income target.
- Correlation analysis among numerical features.
- Analysis of relationships between selected demographic and employment attributes and income.

### EDA Visualizations

The following visualizations were generated and saved in `outputs/graphs/`:

- Income distribution.
- Workclass distribution.
- Education distribution.
- Marital status distribution.
- Occupation distribution.
- Relationship distribution.
- Race distribution.
- Sex distribution.
- Native country distribution.
- Categorical features compared with income.
- Numerical features compared with income.
- Correlation heatmap.

---

## 5. Machine Learning Models

Five machine learning classification models were implemented and evaluated.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model to predict the income category.

### 2. Decision Tree Classifier

Decision Tree Classifier was used to learn decision rules based on the input features and classify individuals into income categories.

### 3. K-Nearest Neighbors (KNN)

KNN was used to classify individuals based on the similarity of their feature values to neighboring observations.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes was implemented using the GaussianNB classifier to predict income categories based on probabilistic assumptions.

### 5. Random Forest Classifier

Random Forest was implemented using multiple decision trees to improve predictive performance and provide feature importance analysis.

---

## 6. Model Evaluation Metrics

The five models were evaluated using the same testing dataset.

The following evaluation metrics were used:

### Accuracy

Accuracy measures the proportion of correctly classified observations among all observations.

### Precision

Precision measures how many of the individuals predicted as belonging to the `>50K` class were actually in that class.

### Recall

Recall measures how many of the actual `>50K` individuals were correctly identified by the model.

### F1-score

F1-score is the harmonic mean of precision and recall and is useful for evaluating the balance between the two metrics.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two income classes across different classification thresholds.

### Matthews Correlation Coefficient (MCC)

MCC measures the correlation between actual and predicted classifications and considers true positives, true negatives, false positives and false negatives.

---

## 7. Model Comparison Results

The performance of the five machine learning models was compared using Accuracy, Precision, Recall, F1-score, ROC-AUC and Matthews Correlation Coefficient.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 85.79% | 74.26% | 62.76% | 0.6803 | 0.9115 | 0.5932 |
| Decision Tree | 82.36% | 63.13% | 64.41% | 0.6376 | 0.7623 | 0.5211 |
| KNN | 83.50% | 66.89% | 62.37% | 0.6455 | 0.8641 | 0.5387 |
| Naive Bayes | 53.93% | 33.86% | 95.66% | 0.5002 | 0.7512 | 0.3335 |
| Random Forest | 85.71% | 73.39% | 63.84% | 0.6828 | 0.9074 | 0.5936 |

---

## 8. Model Observations

### Logistic Regression

Logistic Regression achieved the highest accuracy of 85.79% and ROC-AUC of 0.9115.

It demonstrated strong classification performance and effective discrimination between the two income classes. Its F1-score was also close to the Random Forest model.

### Decision Tree

Decision Tree achieved an accuracy of 82.36% and recall of 64.41%.

It identified a reasonable proportion of individuals belonging to the higher-income class, but its overall predictive performance and ROC-AUC were comparatively lower.

### K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 83.50% and ROC-AUC of 0.8641.

It demonstrated moderate classification performance but did not outperform Logistic Regression or Random Forest across the primary evaluation metrics.

### Naive Bayes

Naive Bayes achieved the highest recall of 95.66% for the higher-income class.

However, its accuracy was 53.93% and precision was 33.86%, indicating that it generated a relatively high number of false-positive predictions.

### Random Forest

Random Forest achieved the highest F1-score of 0.6828 and Matthews Correlation Coefficient of 0.5936.

It demonstrated a strong balance between precision and recall for the higher-income class and performed competitively across the evaluation metrics.

---

## 9. Model Performance Visualizations

The model comparison graphs were generated and saved in `outputs/graphs/`.

The following graphs were created:

- Accuracy comparison.
- Precision comparison.
- Recall comparison.
- F1-score comparison.
- ROC-AUC comparison.
- MCC comparison.

These visualizations provide a graphical comparison of the performance of all five machine learning models.

---

## 10. Random Forest Feature Importance

Feature importance analysis was performed using the trained Random Forest model.

The feature importance values were extracted from the trained classifier after preprocessing and transformation of the input features.

The feature importance values were sorted in descending order to identify the features that contributed most to the model's predictions.

The feature importance data was saved in:

`outputs/feature_importance.csv`

A visualization of the top 15 important features was also generated and saved in:

`outputs/graphs/random_forest_feature_importance.png`

---

## 11. Overall Conclusion

Five machine learning models were trained and evaluated for the Adult Income Classification task.

The models were compared using Accuracy, Precision, Recall, F1-score, ROC-AUC and Matthews Correlation Coefficient.

Random Forest was selected as the best balanced performer based on its highest F1-score of 0.6828 and MCC of 0.5936.

Logistic Regression also demonstrated excellent performance, achieving the highest accuracy of 85.79% and ROC-AUC of 0.9115.

Naive Bayes achieved the highest recall but showed comparatively lower accuracy and precision.

The results demonstrate that model selection depends on the evaluation metric and the objectives of the classification task. Random Forest provides the best balance based on F1-score and MCC in this experiment, while Logistic Regression remains a highly competitive model.

---

## 12. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 13. Project Structure

```text
adult-income-classification/
│
├── data/
│   ├── adult.data
│   └── adult_cleaned.csv
│
├── models/
│   └── random_forest_model.pkl
│
├── outputs/
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── graphs/
│       ├── accuracy_comparison.png
│       ├── precision_comparison.png
│       ├── recall_comparison.png
│       ├── f1_score_comparison.png
│       ├── roc_auc_comparison.png
│       ├── mcc_comparison.png
│       ├── random_forest_feature_importance.png
│       └── EDA visualizations
│
├── src/
│   ├── clean_data.py
│   ├── data_quality.py
│   ├── eda.py
│   ├── final_check.py
│   ├── inspect_dataset.py
│   ├── modeling.py
│   └── test_model.py
│
└── README.md
```

---

## 14. Repository

GitHub Repository: [Adult Income Classification](https://github.com/darshanjadhav-07/adult-income-classification)
