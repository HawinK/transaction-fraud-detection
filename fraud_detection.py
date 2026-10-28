
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and selected threshold
model = joblib.load('fraud_detection_model.pkl')
threshold = 0.63

st.title('Transaction Fraud Detection')
st.markdown(
    'Enter the transaction details below to estimate the likelihood '
    'of fraud.'
)

st.divider()

transaction_type = st.selectbox(
    'Transaction Type',
    ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
)

amount = st.number_input(
    'Transaction Amount',
    min_value=0.0,
    value=1000.0
)

oldbalanceOrg = st.number_input(
    'Old Balance (Sender)',
    min_value=0.0,
    value=10000.0
)

newbalanceOrig = st.number_input(
    'New Balance (Sender)',
    min_value=0.0,
    value=9000.0
)

oldbalanceDest = st.number_input(
    'Old Balance (Receiver)',
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    'New Balance (Receiver)',
    min_value=0.0,
    value=0.0
)

if st.button('Predict', type='primary'):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'balancDiffOrig': [
            oldbalanceOrg - newbalanceOrig
        ],
        'balanceDiffDest': [
            newbalanceDest - oldbalanceDest
        ]
    })

    # Get the estimated probability of fraud
    fraud_probability = model.predict_proba(input_data)[0, 1]

    # Apply the threshold chosen during model evaluation
    prediction = int(fraud_probability >= threshold)

    st.subheader('Prediction Results')
    st.metric(
        'Estimated fraud probability',
        f'{fraud_probability:.2%}'
    )

    st.caption(f'Classification threshold: {threshold:.2f}')

    if prediction == 1:
        st.error('Potential fraud detected.')
    else:
        st.success('Transaction classified as non-fraudulent.')

    st.caption(
        'This is a machine-learning prediction, not confirmation '
        'that a transaction is fraudulent or legitimate.'
    )
