import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive plotting backend

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from streamlit import form
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

df = pd.read_csv('AIML Dataset.csv')
df.head()
# print(df.head())
# df.info()
# df.columns
#df['isFraud'].value_counts()
# df['isFlaggedFraud'].value_counts()
df.isnull().sum().sum()
round((df['isFraud'].value_counts()[1]) / df.shape[0] * 100, 2)


df['type'].value_counts().plot(kind='bar', title='Transaction Type', color='pink')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()

fraud_by_type = df.groupby('type')['isFraud'].mean().sort_values(ascending=False)
fraud_by_type.plot(kind='bar', title='Fraud Rate by Type', color='salmon')
plt.ylabel('Fraud Rate')
plt.show()

df['amount'].describe().astype(int)

plt.figure(figsize=(10, 5))

sns.histplot(
    np.log1p(df['amount'].sample(
        n=min(100_000, len(df)),
        random_state=42
    )),
    bins=100,
    kde=False,
    color='purple'
)

plt.title('Transaction Amount Distribution (log scale)')
plt.xlabel('Log(Amount + 1)')
plt.tight_layout()
plt.show()

plot_df = df.loc[df['amount'] < 50000, ['isFraud', 'amount']]

if len(plot_df) > 100_000:
    plot_df = plot_df.sample(n=100_000, random_state=42)

plt.figure(figsize=(8, 5))
sns.boxplot(data=plot_df, x='isFraud', y='amount')
plt.title('Amount vs isFraud (Transactions < 50,000)')
plt.show()

df['balancDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

frauds_per_step = (
    df.loc[df['isFraud'] == 1]
      .groupby('step')
      .size()
      .sort_index()
)

plt.figure(figsize=(10, 5))
plt.plot(frauds_per_step.index, frauds_per_step.values)
plt.xlabel('Step (Time)')
plt.ylabel('Number of Frauds')
plt.title('Frauds Over Time')
plt.grid(True)
plt.show()


df.drop(columns='step', inplace=True)

# finding top customers
top_senders = df['nameOrig'].value_counts().head(10)
top_receivers = df['nameDest'].value_counts().head(10)

fraud_users = df[df['isFraud']==1]['nameOrig'].value_counts().head(10) # all = 1

fraud_types = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]

sns.countplot(data= fraud_types, x='type', hue='isFraud')
plt.title('Fraud Distribution in Transfer and Cash Out Transactions')
plt.show()

corr_columns = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'balancDiffOrig',
    'balanceDiffDest'
]

corr = df[corr_columns].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
plt.close()

zero_after_transfer = df[
    (df['oldbalanceOrg']> 0) & 
    (df['newbalanceOrig'] == 0)&
    (df['type'].isin(['TRANSFER', 'CASH_OUT']))
    ]

len(zero_after_transfer) 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer   
from sklearn.preprocessing import OneHotEncoder

df_model = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
df_model.head()

categorical = ['type']
numeric = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'balancDiffOrig',
    'balanceDiffDest'
]


y = df_model['isFraud']
x = df_model.drop('isFraud', axis = 1)




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# -----------------------------
# 1. Prepare features and target
# -----------------------------
df_model = df.drop(
    columns=['nameOrig', 'nameDest', 'isFlaggedFraud'],
    errors='ignore'
).copy()

categorical = ['type']

base_numeric = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

engineered_numeric = base_numeric + [
    'balancDiffOrig',
    'balanceDiffDest'
]

y = df_model['isFraud']
x = df_model.drop(columns='isFraud')

print("Total transactions:", len(df_model))
print("Fraudulent transactions:", int(y.sum()))
print("Fraud rate (%):", round(y.mean() * 100, 4))

# -----------------------------
# 2. Split into train, validation and test
# -----------------------------
# 56% training, 14% validation, 30% test
x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y, test_size=0.30, stratify=y, random_state=42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval,
    test_size=0.20, stratify=y_trainval, random_state=42
)

print("Training transactions:", len(y_train))
print("Validation transactions:", len(y_val))
print("Test transactions:", len(y_test))

# -----------------------------
# 3. Create model pipelines
# -----------------------------
def make_pipeline(numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical)
        ],
        remainder='drop'
    )

    return Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(
            class_weight='balanced', max_iter=1000
        ))
    ])

models = {
    'Original features': make_pipeline(base_numeric),
    'Engineered features': make_pipeline(engineered_numeric)
}

# -----------------------------
# 4. Compare models on validation data
# -----------------------------
comparison = []

for name, model in models.items():
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)

    comparison.append({
        'Model': name,
        'ROC-AUC': roc_auc_score(y_val, val_prob),
        'Average precision': average_precision_score(y_val, val_prob),
        'Precision at 0.5': precision_score(
            y_val, val_pred, zero_division=0
        ),
        'Recall at 0.5': recall_score(
            y_val, val_pred, zero_division=0
        )
    })

print("\nVALIDATION MODEL COMPARISON")
print(pd.DataFrame(comparison).round(4).to_string(index=False))

# Select the model with the higher validation average precision
best_name = max(
    models,
    key=lambda name: next(
        row['Average precision']
        for row in comparison if row['Model'] == name
    )
)

pipeline = models[best_name]
print("\nSelected model:", best_name)

# -----------------------------
# 5. Choose threshold on validation data
# -----------------------------
val_prob = pipeline.predict_proba(x_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.01, 1.00, 0.01):
    val_pred = (val_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, val_pred, labels=[0, 1]
    ).ravel()

    threshold_results.append({
        'threshold': round(threshold, 2),
        'precision': precision_score(
            y_val, val_pred, zero_division=0
        ),
        'recall': recall_score(
            y_val, val_pred, zero_division=0
        ),
        'f1': f1_score(
            y_val, val_pred, zero_division=0
        ),
        'false_positives': fp,
        'missed_frauds': fn,
        'frauds_detected': tp
    })

threshold_table = pd.DataFrame(threshold_results)

print("\nVALIDATION THRESHOLD COMPARISON")
print(
    threshold_table[
        threshold_table['threshold'].isin(
            [0.3, 0.5, 0.7, 0.9]
        )
    ].round(4).to_string(index=False)
)

# Prioritise at least 90% recall, then maximise precision
eligible = threshold_table[threshold_table['recall'] >= 0.90]

if not eligible.empty:
    chosen = eligible.loc[eligible['precision'].idxmax()]
else:
    # If no threshold reaches 90% recall, choose the highest-recall option
    chosen = threshold_table.loc[threshold_table['recall'].idxmax()]

best_threshold = float(chosen['threshold'])

print("\nChosen threshold:", best_threshold)
print("Validation precision:", round(chosen['precision'], 4))
print("Validation recall:", round(chosen['recall'], 4))
print("Validation F1:", round(chosen['f1'], 4))

# -----------------------------
# 6. Final evaluation on untouched test data
# -----------------------------
test_prob = pipeline.predict_proba(x_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

print("\nFINAL TEST RESULTS")
print("Chosen threshold:", best_threshold)
print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))
print(
    "Average precision:",
    round(average_precision_score(y_test, test_prob), 4)
)
print("Precision:", round(precision_score(
    y_test, test_pred, zero_division=0
), 4))
print("Recall:", round(recall_score(
    y_test, test_pred, zero_division=0
), 4))
print("F1-score:", round(f1_score(
    y_test, test_pred, zero_division=0
), 4))
print("False positives:", fp)
print("Missed frauds:", fn)
print("Frauds detected:", tp)
print("\nClassification report:")
print(classification_report(y_test, test_pred, zero_division=0))

# -----------------------------
# 7. Save the fitted model
# -----------------------------
joblib.dump(pipeline, 'fraud_detection_model.pkl')
print("\nModel saved to fraud_detection_model.pkl")
