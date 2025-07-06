import streamlit as st
import numpy as np
import joblib
from xgboost import XGBClassifier
import joblib

# Load the trained model
saved = joblib.load("xgb_model_with_encoders.pkl")
model = saved["model"]
le_account = saved["le_account"]
le_chatbot = saved["le_chatbot"]


# --- Page Configuration ---
st.set_page_config(page_title="Paid Customer Prediction", page_icon="🧠", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.title("ℹ️ About App")
    st.markdown("""
    This app predicts whether a user will convert to a **Paid Customer** using a trained **XGBoost** model.

    It uses:
    - Account Type (ENT/SMB)
    - Chatbot Activation
    - Total Clicks

    🧠 Trained with real data  
    🛠 Built with Streamlit
    """)

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🧮 Will the User Convert to a Paid Customer?</h1>", unsafe_allow_html=True)
st.markdown("### 📥 Input User Details:")

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    account_type = st.selectbox("👤 Account Type", options=["ENT", "SMB"])
with col2:
    activated_chatbot = st.selectbox("💬 Activated Chatbot", options=["Y", "N"])

total_clicks = st.number_input("🖱️ Total Clicks", min_value=0, max_value=1000000, value=50, step=1)

# --- Encode inputs ---
account_type_encoded = le_account.transform([account_type])[0]
activated_chatbot_encoded = le_chatbot.transform([activated_chatbot])[0]


# --- Predict ---
if st.button("🚀 Predict Conversion"):
    input_features = np.array([[account_type_encoded, activated_chatbot_encoded, total_clicks]])
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0][1]

    # --- Result ---
    st.markdown("### 🎯 Prediction Result:")
    if prediction == 1:
        st.success(f"✅ This user is **likely to convert** to a paid customer!")
    else:
        st.error(f"❌ This user is **unlikely to convert** to a paid customer.")

  

