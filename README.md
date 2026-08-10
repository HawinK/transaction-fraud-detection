# Transaction Fraud Detection

A machine learning project for detecting potentially fraudulent financial transactions using transaction-level data. The project combines exploratory data analysis, feature engineering, statistical analysis and logistic regression to build a classification pipeline for fraud detection.

## Overview

The aim of this project was to investigate patterns associated with fraudulent transactions and develop a machine learning model capable of classifying transactions as fraudulent or legitimate.

The project covers:

- Exploratory analysis of transaction behaviour
- Feature engineering from account balance data
- Analysis of fraud rates across transaction types
- Data preprocessing and feature scaling
- Logistic regression classification
- Model evaluation using classification metrics
- Saving the trained model for use in a prediction application
- A Streamlit interface for interactive fraud predictions

## Dataset

The dataset contains transaction-level financial data, including:

- `type` — transaction type
- `amount` — transaction amount
- `oldbalanceOrg` — sender's balance before the transaction
- `newbalanceOrig` — sender's balance after the transaction
- `oldbalanceDest` — recipient's balance before the transaction
- `newbalanceDest` — recipient's balance after the transaction
- `isFraud` — target variable indicating whether the transaction was fraudulent

Customer identifiers and the `isFlaggedFraud` field were excluded from the final modelling dataset.

## Exploratory Data Analysis

Several analyses were performed to investigate the structure of the data and identify potential indicators of fraud.

These included:

- Distribution of transaction types
- Fraud rate by transaction type
- Transaction amount distribution
- Relationship between transaction amount and fraud
- Fraud frequency over time
- Correlation between transaction and account-balance variables
- Analysis of balance changes following transfers and cash withdrawals

The analysis showed that fraudulent transactions were concentrated in particular transaction types, motivating the inclusion of transaction type as a categorical model feature.

## Feature Engineering

Two additional features were derived from the account balance information:

```python
df['balancDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
```

These features represent changes in the sender's and recipient's balances and provide additional information about the movement of funds.

The step variable was also investigated as a potential temporal feature. Since exploratory analysis did not show a clear time-dependent pattern in fraud frequency, it was excluded from the final model.

## Modelling

A logistic regression classifier was used as the baseline predictive model.

The preprocessing pipeline consisted of:

Standardising numerical variables using StandardScaler
One-hot encoding the categorical transaction type
Training a logistic regression classifier
Using class_weight='balanced' to account for the imbalance between fraudulent and legitimate transactions

The model was implemented using a Scikit-learn Pipeline and ColumnTransformer, allowing preprocessing and classification to be applied consistently.

```python
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ))
])
```

The data was split into training and test sets using a stratified 70/30 split.

## Results

The model achieved approximately 94.6% accuracy on the test set.

Because fraud detection is an imbalanced classification problem, accuracy alone is not sufficient to assess model performance. Classification metrics and a confusion matrix were therefore also examined to evaluate the model's ability to distinguish fraudulent transactions.

## Interactive Prediction App

The trained model was saved using joblib and integrated into a Streamlit application.

The application allows a user to enter transaction characteristics and receive a prediction indicating whether the transaction is classified as potentially fraudulent.

The application takes:

Transaction type
Transaction amount
Sender's previous balance
Sender's new balance
Recipient's previous balance
Recipient's new balance

and passes these values through the saved preprocessing and classification pipeline.

## Technologies

- Programming: Python

- Data analysis: Pandas, NumPy

- Visualisation: Matplotlib, Seaborn

- Machine learning: Scikit-learn

- Application: Streamlit

- Model persistence: Joblib

## Future Improvements

Potential extensions to the project include:

- Comparing logistic regression with tree-based models such as Random Forest or Gradient Boosting

- Optimising the classification threshold for fraud detection

- Evaluating precision-recall and ROC-AUC curves

- Performing cross-validation and hyperparameter optimisation

- Investigating additional engineered transaction features

- Improving the Streamlit interface and deploying the application

- Exploring more sophisticated approaches to highly imbalanced classification
