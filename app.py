import streamlit as st
import pandas as pd


st.set_page_config(page_title="Temperature Prediction App", layout="wide")

st.title("🌡️ Temperature Prediction App")
st.markdown("Welcome to the Temperature Prediction App for Beijing City!")

# Sidebar with Header and Radio Buttons
with st.sidebar:
    st.header("📂 Pages")
    page = st.radio("Navigate to:", ["Dataset Overview", "Explore EDA", "Model Prediction"])

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("beijing_cleaned.csv")

data = load_data()

# Page Navigation Logic
if page == "Dataset Overview":
    st.subheader("📄 Dataset Overview")
    st.dataframe(data)
    st.write("Shape:", data.shape)

elif page == "Explore EDA":
    st.subheader("📊 Exploratory Data Analysis")
    st.write(data.describe())

elif page == "Model Prediction":
    st.subheader("🤖 Model Prediction")
    st.info("This section will include machine learning models (coming soon).")






    
    
