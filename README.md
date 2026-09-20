💳 Credit Card Fraud Detection using Machine Learning

📌 Project Overview

This project is a Machine Learning based Credit Card Fraud Detection system designed to identify potentially fraudulent transactions.

The system uses a Random Forest Classifier and provides an interactive Streamlit web application where users can upload transaction data and receive fraud predictions, fraud probabilities, and risk levels.



🎯 Objectives

- Detect fraudulent credit card transactions
- Handle highly imbalanced transaction data
- Perform exploratory data analysis
- Train and evaluate Machine Learning models
- Provide an interactive fraud detection dashboard
- Classify transactions based on risk level



🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit
- Jupyter Notebook



 🤖 Machine Learning

 Models Used

- Logistic Regression
- Random Forest Classifier

 Data Preprocessing

- Duplicate removal
- Missing-value analysis
- Feature scaling using StandardScaler
- Handling class imbalance using SMOTE
- Stratified train-test split

 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix



 📊 Dataset

The project uses the Credit Card Fraud Detection dataset containing credit card transactions with anonymized features.

The dataset contains:

- `Time`
- `V1` – `V28`
- `Amount`
- `Class`

The original CSV dataset is not included in this repository because of its large file size.



 🚀 Streamlit Application

The project includes an interactive Streamlit application.

 Features

- 📂 CSV file upload
- 📊 Dataset preview
- 🚨 Fraud prediction
- 📈 Fraud probability
- ⚠️ Risk-level classification
- 📋 Transaction prediction table
- 📊 Risk distribution visualization

 Risk Levels

| Fraud Probability | Risk Level |
|---|---|
| < 0.30 | Low Risk |
| 0.30 – 0.69 | Medium Risk |
| ≥ 0.70 | High Risk |



 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app/
│   └── app.py
│
├── data/
│   └── Dataset not included
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── .gitignore
└── README.md
