import streamlit as st
import os

st.set_page_config(page_title="🌡️ Temperature Prediction App", layout="wide")

# App title and intro
st.title("👋 Welcome to the 🌡️ Temperature Prediction App")
st.subheader("Explore Beijing's air quality and predict temperatures based on pollutant levels.")

# Navigation radio button
pages = {
    "📄 Dataset Overview": "pages/1_Dataset_Overview.py",
    "📊 Explore EDA": "pages/2_Explore_EDA.py",
    "📈 Model Prediction": "pages/3_Model_Prediction.py"
}

page_choice = st.radio("Navigate to:", list(pages.keys()))

# Redirect to selected page
st.switch_page(pages[page_choice])












   









   















    
    
