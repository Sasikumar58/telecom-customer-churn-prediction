# Telecom Customer Churn Prediction

A machine learning project that predicts telecom customer churn and
identifies customer characteristics associated with higher churn risk.

## Project Overview

Customer churn is an important business problem for telecom companies
because retaining existing customers can be more efficient than
acquiring new ones.

This project uses customer demographic, service, contract, billing, and
account information to build a machine learning pipeline for predicting
whether a customer is likely to churn.

## Business Problem

The objective is to:

-   Predict customers who are likely to churn.
-   Identify factors associated with customer churn.
-   Evaluate multiple machine learning models.
-   Handle class imbalance using SMOTE.
-   Build a reusable prediction pipeline.
-   Provide results that can support customer-retention strategies.

## Dataset

The project uses the Telco Customer Churn dataset.

-   **Customers:** 7,043
-   **Features:** 21 columns
-   **Target:** `Churn`
-   **Overall churn rate:** 26.54%

The `customerID` column is removed because it is an identifier and does
not provide useful predictive information.

`TotalCharges` is converted to numeric format during preprocessing.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Imbalanced-learn
-   XGBoost
-   Jupyter Notebook

## Machine Learning Workflow

``` text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
Feature Preprocessing
     ↓
SMOTE for Class Imbalance
     ↓
Model Training
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Test Evaluation
     ↓
Permutation Importance
     ↓
Churn Prediction Pipeline
```

## Exploratory Data Analysis

The project analyzes churn across several important customer attributes.

### Contract Type

  Contract           Churn Rate
  ---------------- ------------
  Month-to-month         42.71%
  One year               11.27%
  Two year                2.83%

The analysis shows a substantial difference in observed churn rates
across contract types.

### Internet Service

  Internet Service     Churn Rate
  ------------------ ------------
  Fiber optic              41.89%

### Payment Method

Customers using electronic check had an observed churn rate of:

**45.29%**

### Support Services

Customers without Tech Support had an observed churn rate of:

**41.64%**

Customers without Online Security had an observed churn rate of:

**41.77%**

These are descriptive findings from the dataset and should not be
interpreted as proof that a particular service directly causes churn.

## Models Evaluated

The project evaluates:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest
4.  XGBoost

Model comparison uses cross-validation and classification metrics
including:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC

## Class Imbalance

The churn target is imbalanced because churned customers represent
26.54% of the dataset.

SMOTE is incorporated inside the machine-learning pipeline so that
oversampling is performed appropriately during model validation rather
than before cross-validation.

## Final Model Evaluation

A tuned Random Forest model was evaluated on the untouched test set.

  Metric        Test Result
  ----------- -------------
  Accuracy            75.9%
  Precision           53.6%
  Recall              67.6%
  F1-score            59.8%
  ROC-AUC             82.9%

The test-set results provide a more realistic estimate of how the
trained pipeline performs on previously unseen customers.

## Feature Importance

Permutation-importance analysis was used to investigate which features
contributed most to the model's predictions.

The strongest feature identified in the notebook was:

**Contract**

Other important features included:

-   Online Security
-   Tech Support

Feature importance indicates association with model predictions; it does
not establish causation.

## Reusable Prediction Pipeline

The project saves the trained preprocessing and prediction pipeline as:

``` text
telecom_customer_churn_pipeline.pkl
```

This makes it possible to reuse the same preprocessing and model logic
when predicting churn for new customers.

## Example Prediction Workflow

The notebook includes a customer-level prediction function that returns:

-   Predicted churn class
-   Churn probability

This provides a foundation for integrating the model into an application
or dashboard.

## Project Structure

``` text
telecom-customer-churn-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── telecom_customer_churn_prediction.ipynb
│
├── models/
│   └── telecom_customer_churn_pipeline.pkl
│
└── data/
    └── README.md
```

## How to Run

### 1. Clone the repository

``` bash
git clone https://github.com/Sasikumar58/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Open the notebook

``` bash
jupyter notebook
```

Then open:

``` text
notebooks/telecom_customer_churn_prediction.ipynb
```

### 4. Run the notebook

Run all cells from beginning to end.

## Key Skills Demonstrated

-   Data cleaning
-   Exploratory data analysis
-   Feature preprocessing
-   Categorical feature encoding
-   Train/test splitting
-   Class-imbalance handling
-   SMOTE
-   Machine learning model comparison
-   Cross-validation
-   Hyperparameter tuning
-   Classification metrics
-   Confusion matrix
-   ROC-AUC analysis
-   Permutation importance
-   Model pipeline creation
-   Model serialization with pickle
-   Business-oriented interpretation

## Future Improvements

Potential next steps for the project include:

-   Build a Power BI dashboard.
-   Deploy the model using Streamlit.
-   Add automated model monitoring.
-   Experiment with additional hyperparameter optimization.
-   Add explainability tools such as SHAP.
-   Create a customer-retention recommendation layer.

## Interview Summary

**60-second explanation:**

> I developed a telecom customer churn prediction system using Python
> and machine learning. I first cleaned and explored the Telco Customer
> Churn dataset, then analyzed churn patterns across contract type,
> internet service, payment method, and support services. Since the
> target variable was imbalanced, I used SMOTE within the
> machine-learning pipeline. I compared Logistic Regression, Decision
> Tree, Random Forest, and XGBoost models using cross-validation and
> classification metrics. I then tuned a Random Forest model and
> evaluated it on an untouched test set, achieving 75.9% accuracy and
> 82.9% ROC-AUC. Finally, I used permutation importance to understand
> important features and saved the complete preprocessing and prediction
> pipeline so it can be reused for new customers.

## Author

**Sasikumar**

GitHub: `Sasikumar58`

------------------------------------------------------------------------

This project demonstrates an end-to-end machine learning workflow for a
practical business problem, from data preparation and exploratory
analysis through model development, evaluation, interpretation, and
reusable prediction.
