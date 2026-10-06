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
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
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

Data Preprocessing
The following preprocessing steps were performed:
1. Inspected the dataset structure and data types.
2. Handled missing values in TotalCharges.
3. Removed customerID from predictive features.
4. Encoded categorical variables using LabelEncoder.
5. Separated features and target variable.
6. Divided the dataset into training and testing sets using an 80:20 split.
7. Applied SMOTE to the training data to address class imbalance.
The test set was kept separate and was not oversampled.
Machine Learning Models
Three classification algorithms were evaluated:
Decision Tree
A supervised learning algorithm that recursively divides the dataset using feature-based decision rules to classify customers into churn and non-churn categories.
Random Forest
An ensemble learning algorithm that combines multiple decision trees and uses their collective predictions to improve classification performance and generalization.
XGBoost
A gradient boosting algorithm that builds decision trees sequentially, with each new tree attempting to reduce the errors made by previous trees.
Model Comparison
The models were evaluated using stratified 5-fold cross-validation with SMOTE applied within each training fold.
Model	Mean CV Accuracy	Standard Deviation
Decision Tree	71.37%	0.76%
Random Forest	77.51%	1.06%
XGBoost	76.64%	0.89%


Random Forest achieved the highest mean cross-validation accuracy among the three evaluated models.
Hyperparameter Tuning
Random Forest was further optimized using RandomizedSearchCV.
The selected configuration was:
Parameter	Value
n_estimators	100
max_depth	None
min_samples_split	2
min_samples_leaf	2
max_features	log2


The tuned model achieved a cross-validation accuracy of approximately 77.76%.
Final Model Performance
The tuned Random Forest model was evaluated on the unseen test dataset.
Metric	Result
Accuracy	77.50%
Churn Precision	57%
Churn Recall	62%
Churn F1-Score	60%
ROC-AUC	0.8330


Confusion Matrix
The final model produced the following confusion matrix:
	Predicted No Churn	Predicted Churn
Actual No Churn	859	177
Actual Churn	140	233


The model correctly identified 233 churn customers while producing 140 false negatives.
Streamlit Application
The project includes an interactive Streamlit dashboard with three main sections.
1. Customer Churn Prediction
Users can enter:
- Demographic information
- Service details
- Contract information
- Payment method
- Monthly charges
- Total charges
- Customer tenure
The application then provides:
- Predicted customer status
- Churn probability
- Churn risk interpretation
2. Model Performance
The dashboard provides interactive visualisations for:
- Model comparison
- Confusion matrix
- ROC curve
- Accuracy
- Churn recall
- Churn F1-score
- ROC-AUC
3. Customer Analytics
The dashboard provides interactive analysis of the original customer dataset, including:
- Customer churn distribution
- Churn rate by contract type
- Churn rate by customer tenure
- Churn rate by internet service
- Churn rate by payment method
The visualisations support interactive filtering and hover-based exploration.
Project Structure
Customer-Churn-Prediction/
│
├── app.py
├── Customer_Churn_ML_Project.ipynb
│
├── customer_churn_model.pkl
├── encoders.pkl
├── roc_data.pkl
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── README.md
└── .gitignore

File Description
File	Description
app.py	Streamlit application
Customer_Churn_ML_Project.ipynb	Complete machine learning workflow
customer_churn_model.pkl	Saved tuned Random Forest model
encoders.pkl	Saved categorical encoders
roc_data.pkl	ROC curve data generated from the final model
WA_Fn-UseC_-Telco-Customer-Churn.csv	Telco customer churn dataset
README.md	Project documentation
.gitignore	Git configuration for excluding unnecessary files


Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Plotly
- Streamlit
- Jupyter Notebook
Installation
Clone the repository:
git clone https://github.com/Anusha200513/Customer-Churn-Prediction.git

Navigate to the project directory:
cd Customer-Churn-Prediction

Install the required dependencies:
pip install streamlit pandas numpy scikit-learn imbalanced-learn xgboost matplotlib plotly

Running the Application
Start the Streamlit application using:
streamlit run app.py

The application will open in the browser and provide access to the prediction, model performance, and customer analytics sections.
Key Outcome
The project demonstrates an end-to-end machine learning workflow for customer churn prediction, from exploratory data analysis and preprocessing to model evaluation and interactive deployment.
The final tuned Random Forest model achieved:
77.50% test accuracy
and
0.8330 ROC-AUC
while providing an interactive interface for generating predictions on new customer information.
Future Scope
Potential improvements include:
- Feature importance and explainable AI visualisations
- SHAP-based individual prediction explanations
- Probability threshold optimization for better churn recall
- Integration with a customer relationship management system
- Automated model retraining using new customer data
- Cloud deployment of the Streamlit application
References
1. IBM Telco Customer Churn Dataset.
2. Scikit-learn documentation.
3. Imbalanced-learn documentation.
4. XGBoost documentation.
5. Streamlit documentation.

 
