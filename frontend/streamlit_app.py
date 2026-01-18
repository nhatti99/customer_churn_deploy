import streamlit as st
import requests

st.title('Bank Churn Prediction')

st.write('Enter customer details:')

credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.number_input('Age', min_value=18, max_value=100, value=35)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0.0, value=50000.0)
num_of_products = st.number_input('Num Of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credict Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=100000.0)

if st.button('Predict Churn'):
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    response = requests.post('https://customer-churn-deploy-3vje.onrender.com/predict', json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Churn Probability: {result['churn_probability']:.2f}")
        st.write('Churned:' if result['churned'] else 'Not Churned')
    else:
        st.error('Prediction failed. Please check the API.')
