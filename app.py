import streamlit as st

# App title and welcome message
st.set_page_config(page_title="🌡️ Temperature Prediction App", layout="centered")

st.markdown("## 👋 Welcome!")
st.title("🌡️ Temperature Prediction App")
st.subheader("Welcome to the Temperature Prediction App for Beijing City!")

# Radio button navigation
pages = st.radio(
    "Navigate to a page:",
    ("Home", "Dataset Overview", "Explore EDA", "Model Prediction"),
    index=0
)

# Redirecting to selected page
if page == "Dataset Overview":
    st.switch_page("pages/Dataset_Overview.py")
elif page == "Explore EDA":
    st.switch_page("pages/Explore_EDA.py")
elif page == "Modeling":
    st.switch_page("pages/Model_Prediction.py")












   









   















    
    
