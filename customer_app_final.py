#!/usr/bin/env python
# coding: utf-8

# In[22]:


import streamlit as st
import pandas as pd
import joblib
import base64
import os

# ✅ Set page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

# ✅ Function to set background image (without lightening it)
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main-content {{
            background-color: rgba(255, 255, 255, 0.8);  /* semi-transparent block for readability */
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Apply background image
if os.path.exists("mall_background1.png"):
    set_background("mall_background1.png")
else:
    st.warning("⚠️ Background image not found: 'mall_background1.png'")

# ✅ Load model and scaler
model_loaded = True
try:
    kmeans = joblib.load("kmeans_model1.pkl")
    scaler = joblib.load("scaler1.pkl")
except Exception as e:
    model_loaded = False
    st.error("🚫 Failed to load model or scaler.")
    st.exception(e)

# ✅ UI content within styled container
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("🛍️ Customer Segmentation")
    st.markdown("### 👤 Predict Customer Segment")

    age = st.slider("Select Age", 18, 70, 30)
    income = st.slider("Select Annual Income (k$)", 15, 150, 60)
    score = st.slider("Select Spending Score (1–100)", 1, 100, 50)

    if st.button("Predict Segment") and model_loaded:
        try:
            input_df = pd.DataFrame([[age, income, score]], columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
            input_scaled = scaler.transform(input_df)
            cluster = kmeans.predict(input_scaled)[0]

            st.success(f"🎯 The customer belongs to **Segment {cluster}**")

            segment_descriptions = {
                0: "📈 *Loyal Big Spenders* — Highly engaged and profitable. Focus on VIP retention strategies.",
                1: "🛍️ *Mid-Range Buyers* — Steady and reliable. Encourage upsells and targeted promotions.",
                2: "🔍 *Price-Sensitive Shoppers* — Value-conscious. Attract with deals, loyalty points, and discounts."
            }
            st.info(f"🧠 Segment Insight: {segment_descriptions.get(cluster, 'No description available')}")

        except Exception as e:
            st.error("🚫 Prediction failed. Please check the input or model files.")
            st.exception(e)

    st.markdown("---")
    st.markdown("<center>Developed with ❤️ by <b>Bhavpreet Kaur</b> for Smart Marketing Strategy</center>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# In[ ]:




