# import streamlit as st
# import pandas as pd
# import joblib
# import os
from Src.model import FraudPipeline, FeatureEngineering, Preprocessing, LogTransformer

# # Load model
# MODEL_PATH = os.path.join("Notebooks", "artifacts", "fraud_pipeline.pkl")
# fp_loaded = joblib.load(MODEL_PATH)

# # Streamlit App
# st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

# # Title and Problem Statement
# st.title("🔍 Fraud Detection App")

# st.markdown("""
# ### Problem Statement  
# Develop a machine learning model to detect **potentially fraudulent transactions** based on features like account age, payment method, time of transaction, and category.  
# The model classifies transactions as either:
# - **Legitimate (0)**
# - **Potentially Fraudulent (1)**
# ---
# ### Input Feature Descriptions
# - **accountAgeDays**: Number of days the account has been active  
# - **numItems**: Number of items in the transaction  
# - **localTime**: Time of transaction (float, possibly in hours)  
# - **paymentMethod**: Method of payment (PayPal, store credit, credit card)  
# - **paymentMethodAgeDays**: Number of days since the payment method was linked to the account  
# - **isWeekend**: 1 if transaction occurred on weekend, 0 otherwise  
# - **Category**: Category of transaction (e.g., electronics, shopping, food)  
# ---
# """)

# # Initialize session state for auto-fill
# if "inputs" not in st.session_state:
#     st.session_state.inputs = {
#         "Category": "shopping",
#         "paymentMethod": "paypal",
#         "isWeekend": 0,
#         "numItems": 1,
#         "localTime": 4.742303,
#         "paymentMethodAgeDays": 0,
#         "accountAgeDays": 1
#     }

# # Auto-fill buttons
# col1, col2 = st.columns(2)
# with col1:
#     if st.button("Load Legitimate Example"):
#         st.session_state.inputs = {
#             "Category": "shopping",
#             "paymentMethod": "paypal",
#             "isWeekend": 0,
#             "numItems": 4,
#             "localTime": 4.742303,
#             "paymentMethodAgeDays": 0,
#             "accountAgeDays": 4000
#         }
# with col2:
#     if st.button("Load Fraudulent Example"):
#         st.session_state.inputs = {
#             "Category": "shopping",
#             "paymentMethod": "paypal",
#             "isWeekend": 0,
#             "numItems": 4,
#             "localTime": 4.742303,
#             "paymentMethodAgeDays": 0,
#             "accountAgeDays": 1
#         }

# # Input fields with default or loaded values
# category = st.selectbox(
#     "Category (e.g., electronics, shopping, food, other)",
#     ["shopping", "food", "electronics", "other"],
#     index=["shopping", "food", "electronics", "other"].index(st.session_state.inputs["Category"])
# )

# payment_method = st.selectbox(
#     "Payment Method (PayPal, creditcard, debitcard)",
#     ["paypal", "creditcard", "debitcard"],
#     index=["paypal", "creditcard", "debitcard"].index(st.session_state.inputs["paymentMethod"])
# )

# is_weekend = st.selectbox(
#     "Is Weekend? (1 = Yes, 0 = No)",
#     [0, 1],
#     index=[0, 1].index(st.session_state.inputs["isWeekend"])
# )

# num_items = st.number_input(
#     "Number of Items (count of items in transaction)",
#     min_value=1,
#     value=st.session_state.inputs["numItems"]
# )

# local_time = st.number_input(
#     "Local Time (float: e.g., 4.742303)",
#     value=float(st.session_state.inputs["localTime"]),
#     format="%.6f"
# )

# payment_age = st.number_input(
#     "Payment Method Age (days since method linked)",
#     min_value=0,
#     value=st.session_state.inputs["paymentMethodAgeDays"]
# )

# account_age = st.number_input(
#     "Account Age (days since account created)",
#     min_value=0,
#     value=st.session_state.inputs["accountAgeDays"]
# )

# # Prediction logic
# if st.button("Predict Fraud"):
#     # Prepare input data
#     input_data = pd.DataFrame([{
#         "Category": category,
#         "paymentMethod": payment_method,
#         "isWeekend": is_weekend,
#         "numItems": num_items,
#         "localTime": local_time,
#         "paymentMethodAgeDays": payment_age,
#         "accountAgeDays": account_age
#     }])

#     # Get fraud probability
#     fraud_prob = fp_loaded.predict_proba(input_data)[0]  # probability of fraud

#     # Apply threshold tuning
#     prediction = 1 if fraud_prob >= fp_loaded.best_threshold else 0

#     # Show result with confidence
#     if prediction == 1:
#         st.error(f"🚨 Fraudulent Transaction!\nConfidence: {fraud_prob*100:.2f}%")
#     else:
#         st.success(f"✅ Legitimate Transaction\nConfidence: {(1-fraud_prob)*100:.2f}%")

# # Example outcome explanation
# st.markdown("""
# ---
# ### Example Outcomes:
# #### Legitimate Example (0 ✅ Legitimate Transaction)
# {
# "Category": "shopping",
# "paymentMethod": "paypal",
# "isWeekend": 0,
# "numItems": 4,
# "localTime": 4.742303,
# "paymentMethodAgeDays": 0,
# "accountAgeDays": 4000
# }
# #### Fraudulent Example (1 🚨 Fraudulent Transaction!)
# {
# "Category": "shopping",
# "paymentMethod": "paypal",
# "isWeekend": 0,
# "numItems": 4,
# "localTime": 4.742303,
# "paymentMethodAgeDays": 0,
# "accountAgeDays": 1
# }
# """)


import streamlit as st
from home import home_page
from about_model import about_page
from metrics_page import metrics_page

# Register pages with st.Page
home_pg = st.Page(home_page, title="Home", icon="🏠", default=True)
about_pg = st.Page(about_page, title="About Model", icon="ℹ️")
metrics_pg = st.Page(metrics_page, title="Metrics", icon="📊")

# Navigation
pg = st.navigation([home_pg, about_pg, metrics_pg])
pg.run()