import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("🚀 Startup Success Predictor")
st.markdown("Enter startup details to predict the chance of **acquisition success**.")

# Input form
with st.form("input_form"):
    category_code = st.number_input("Category Code (as number)", min_value=0)
    state_code = st.number_input("State Code (as number)", min_value=0)
    city = st.number_input("City (as number)", min_value=0)
    funding_total_usd = st.number_input("Total Funding (USD)", min_value=0.0)
    funding_rounds = st.number_input("Number of Funding Rounds", min_value=0)
    founded_year = st.number_input("Founded Year", min_value=1950, max_value=2025)
    has_VC = st.selectbox("Has Venture Capital?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# Handle form submission
if submitted:
    has_VC_val = 1 if has_VC == "Yes" else 0
    input_data = np.array([[category_code, state_code, city, funding_total_usd, funding_rounds, founded_year, has_VC_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ This startup has a **high chance of success (acquisition)**!")
    else:
        st.error("⚠️ This startup is **less likely to succeed** based on the data.")
