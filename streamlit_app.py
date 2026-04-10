# streamlit_app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Expense Classifier")

st.title("💰 Expense Account Classifier")

# ---- inputs ----
vendor = st.text_input("Vendor ID", "lL1pcuEf3q6ufBVg2R75")
name = st.text_input("Item Name", "Slack subscription")
desc = st.text_input("Description", "Slack monthly subscription")
amount = st.number_input("Amount (SGD)", value=1000.0)

# ---- predict ----
if st.button("Predict"):

    payload = {
        "vendorId": vendor,
        "itemName": name,
        "itemDescription": desc,
        "itemTotalAmount": amount,
    }

    try:
        res = requests.post(API_URL, json=payload)

        if res.status_code == 200:
            result = res.json()

            st.success(f"Prediction: {result['accountName']}")

        else:
            st.error(f"API Error: {res.text}")

    except Exception as e:
        st.error(f"Connection error: {e}")
