# Customer Churn Prediction Using Machine Learning

A machine learning project that predicts whether a telecommunications customer is likely to churn based on demographic, service, contract, and billing information.

The project includes exploratory data analysis, data preprocessing, class imbalance handling using SMOTE, machine learning model comparison, hyperparameter tuning, evaluation on unseen data, and an interactive Streamlit dashboard for real-time predictions and customer analytics.

---

## Project Overview

Customer churn occurs when an existing customer stops using a company's products or services. Predicting churn can help organizations identify customers who may leave and support proactive customer retention strategies.

This project develops a binary classification system that predicts:

- **Churn** — Customer is likely to leave
- **No Churn** — Customer is likely to remain

The system uses the Telco Customer Churn dataset and evaluates multiple machine learning algorithms before selecting a tuned Random Forest model as the final model.

---

## Problem Statement

To develop a machine learning system that predicts whether a telecommunications customer is likely to churn based on their demographic, service, account, and billing information.

---

## Dataset

The project uses the **Telco Customer Churn dataset**, containing:

- **7,043 customer records**
- **21 original attributes**
- Demographic information
- Service-related information
- Contract information
- Billing information
- Customer churn status

The `customerID` attribute is treated as an identifier and is not used as a predictive feature.

### Important preprocessing observation

The `TotalCharges` column contained 11 blank values. These records corresponded to customers with zero tenure, and the blank values were converted to `0.0`.

---

## Machine Learning Workflow

The project follows the following workflow:

```text
Telco Customer Churn Dataset
            |
            v
Exploratory Data Analysis
            |
            v
Missing Value Handling
            |
            v
Feature Selection
            |
            v
Categorical Encoding
            |
            v
Train-Test Split (80:20)
            |
            v
SMOTE on Training Data
            |
            v
Model Training
            |
      +-----+-----+
      |     |     |
      v     v     v
 Decision  Random  XGBoost
  Tree     Forest
      |     |     |
      +-----+-----+
            |
            v
5-Fold Cross-Validation
            |
            v
Random Forest Hyperparameter Tuning
            |
            v
Final Model Evaluation
            |
            v
Streamlit Deployment
