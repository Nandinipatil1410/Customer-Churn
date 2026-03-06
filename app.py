import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthlyCharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

predictButton = st.button("Predict!")

if predictButton:

    genderSelected = 1 if gender == "Female" else 0

    # Use a DataFrame with column names to match the scaler's expected input
    input_df = pd.DataFrame(
        [[age, genderSelected, tenure, monthlyCharge]],
        columns=["Age", "Gender", "Tenure", "MonthlyCharges"],
    )
    X_scaled = scaler.transform(input_df)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    predicted = "Yes" if prediction == 1 else "No"
    confidence = probabilities[prediction] * 100

    st.balloons()

    if prediction == 1:
        st.error(f"⚠️ Churn Predicted (Confidence: {confidence:.1f}%)")
    else:
        st.success(f"✅ No Churn Predicted (Confidence: {confidence:.1f}%)")

else:
    st.write("Please enter the values and use predict button")
