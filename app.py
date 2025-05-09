import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Temperature Prediction App", page_icon="🌡️", layout="centered")

# --- Title and Welcome ---
st.markdown("<h1 style='text-align: center;'>🌡️ Temperature Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>👋 Welcome to the Temperature Prediction App for Beijing City!</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- Navigation using Radio Buttons ---
st.markdown("### 📂 Navigate to:")
selected_page = st.radio(
    "Choose a section to explore:",
    ["📄 Dataset Overview", "📊 Exploratory Data Analysis", "🤖 Model Prediction"],
    horizontal=True
)

# --- Redirection based on selection ---
if selected_page == "📄 Dataset Overview":
    st.switch_page("pages/Dataset_Overview.py")
elif selected_page == "📊 Exploratory Data Analysis":
    st.switch_page("pages/Explore_EDA.py")
elif selected_page == "📈 Model Prediction":
    st.switch_page("pages/Model_Prediction.py")







   









   















    
    
