import joblib
import numpy as np

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

import streamlit as st

# ye title h mere project ka  

st.title("Creditworthiness Prediction System")

st.write("Enter user financial details to predict creditworthiness")

# yaha pr hmne user se input liya 

income = st.number_input("Enter Income", min_value=0)
debt = st.number_input("Enter Debt", min_value=0)
age = st.number_input("Enter Age", min_value=18, max_value=100)
payment_history = st.selectbox("Payment History", [0, 1])
loan_amount = st.number_input("Enter Loan Amount", min_value=0)

# prediction button ka use kiya

if st.button("Predict"):

    input_data = np.array([[income, debt, payment_history, age, loan_amount]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("You are Creditworthy ✅")
    else:
        st.error("You are Not Creditworthy ❌")
