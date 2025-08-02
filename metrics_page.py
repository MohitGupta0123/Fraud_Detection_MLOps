import streamlit as st
import pandas as pd

def metrics_page():
    st.title("📊 Model Metrics")

    # Replace with your real metrics
    metrics = {
        "Accuracy": 0.95,
        "Precision": 0.82,
        "Recall": 0.78,
        "F1-Score": 0.80,
        "ROC-AUC": 0.91
    }
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

    st.markdown("""
    **Note:** Metrics are based on test set after applying SMOTE and threshold tuning.
    """)

    st.page_link("about_model.py", label="About Model", icon="ℹ️")
    st.page_link("home.py", label="Back to Prediction", icon="🏠")
