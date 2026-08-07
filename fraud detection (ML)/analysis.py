import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from streamlit import form
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

df = pd.read_csv('fraud detection (ML)/AIML Dataset.csv')
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

sns.histplot(np.log1p(df['amount']), bins=100, kde=True, color='purple')
plt.title('Transaction Amount Distribution (log scale)')
plt.xlabel('Log(Amount + 1)')
plt.show()

sns.boxplot(data = df[df['amount'] < 50000], x='isFraud', y='amount')
plt.title('Amount vs isFraud (Transactions < 50,000)')
plt.show()

df['balancDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']


df.head(2)

frauds_per_step = df[df['isFraud'] == 1]['step'].value_counts().sort_index()
plt.plot(frauds_per_step.index, frauds_per_step.values, label='Frauds per Step')
plt.xlabel('Step (Time)')
plt.ylabel('Number of Frauds')
plt.title('Frauds Over Time')
plt.grid(True)
plt.show() # doesn't appear time dependent so not going to use step as a feature

df.drop(columns='step', inplace=True)

# finding top customers
top_senders = df['nameOrig'].value_counts().head(10)
top_receivers = df['nameDest'].value_counts().head(10)

fraud_users = df[df['isFraud']==1]['nameOrig'].value_counts().head(10) # all = 1

fraud_types = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]

sns.countplot(data= fraud_types, x='type', hue='isFraud')
plt.title('Fraud Distribution in Transfer and Cash Out Transactions')
plt.show()

corr = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balancDiffOrig', 'balanceDiffDest']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

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
numeric = ['amount','oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

y = df_model['isFraud']
x = df_model.drop('isFraud', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify = y)

preprocessor = ColumnTransformer(

transformers = [
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop = 'first'), categorical)
    ],
    remainder = 'drop'
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

print(classification_report(y_test, y_pred))

confusion_matrix(y_test, y_pred)

pipeline.score(x_test, y_test) * 100 # 94.6...

import joblib
joblib.dump(pipeline, 'fraud_detection_model.pkl')
