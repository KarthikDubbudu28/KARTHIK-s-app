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
