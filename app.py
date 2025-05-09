import streamlit as st
import pandas as pd

st.title('Welcome to the Temperature Prediction App🌡️')
@st.cache_data
def load_data():
    df = pd.read_csv("beijing_cleaned.csv")
    return df

# Call the load function
data = load_data()

# Display the dataset
st.write("### Here's the cleaned Beijing dataset:")
st.dataframe(data)
