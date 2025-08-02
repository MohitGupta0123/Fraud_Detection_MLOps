import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from Src.model import FraudPipeline, FeatureEngineering, Preprocessing, LogTransformer
# Load model
MODEL_PATH = os.path.join("Notebooks", "artifacts", "fraud_pipeline.pkl")
print(MODEL_PATH)
fp_loaded = joblib.load(MODEL_PATH)
print(fp_loaded)

# Streamlit App
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

st.title("🔍 Fraud Detection App")
st.write("Enter transaction details below to check if it is **fraudulent** or not.")

# Input fields
category = st.selectbox("Category", ["shopping", "food", "electronics", "other"])
payment_method = st.selectbox("Payment Method", ["paypal", "creditcard", "debitcard"])
is_weekend = st.selectbox("Is Weekend?", [0, 1])
num_items = st.number_input("Number of Items", min_value=1, value=1)
local_time = st.number_input("Local Time (float)", value=4.742303, format="%.6f")
payment_age = st.number_input("Payment Method Age (days)", min_value=0, value=0)
account_age = st.number_input("Account Age (days)", min_value=0, value=1)

if st.button("Predict Fraud"):
    # Prepare input data
    input_data = pd.DataFrame([{
        "Category": category,
        "paymentMethod": payment_method,
        "isWeekend": is_weekend,
        "numItems": num_items,
        "localTime": local_time,
        "paymentMethodAgeDays": payment_age,
        "accountAgeDays": account_age
    }])

    # Predict
    prediction = fp_loaded.predict_pipeline(input_data, use_optimal_threshold=True)

    # Show result
    result = "🚨 Fraudulent Transaction!" if prediction[0] == 1 else "✅ Legitimate Transaction"
    st.subheader(result)
