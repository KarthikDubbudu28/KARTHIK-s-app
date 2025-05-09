import streamlit as st
import pandas as pd

st.title('Welcome to the Temperature Prediction App🌡️')
# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("beijing_cleaned.csv")  # File must be in the same directory
    return df

try:
    data = load_data()
    st.write("### Here's the cleaned Beijing dataset:")
    st.dataframe(data)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
with st.sidebar:
    st.header('Pages')
    page = st.radio("Navigate to:", ["Dataset Overview", "Explore EDA", "Model Prediction"])

if page == "Dataset Overview":
    st.subheader("📄 Dataset Overview")
    st.info("This section shows the overview of the data")
   
    
elif page == "Explore EDA":
    st.subheader("📊 Exploratory Data Analysis")
    st.info("This section explores you through EDA")

elif page == "Model Prediction":
    st.subheader("🤖 Model Prediction")
    st.info("Here You Can Predict the Temperature using different models")



    
    
