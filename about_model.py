import streamlit as st

def about_page():
    st.title("ℹ️ About the Model")

    st.markdown("""
    This fraud detection model was trained with:
    - **SMOTE**: To balance fraud vs legitimate transactions
    - **Threshold Tuning**: To maximize F1 and recall
    - **Pipeline**: Feature engineering + preprocessing + classifier

    **Key Features Used:**
    - accountAgeDays, numItems, localTime
    - paymentMethod, paymentMethodAgeDays, isWeekend, Category

    The model predicts:
    - **0 → Legitimate Transaction**
    - **1 → Fraudulent Transaction**

    ---
    """)
    st.page_link("metrics_page.py", label="Go to Metrics", icon="📊")
    st.page_link("home.py", label="Back to Prediction", icon="🏠")
