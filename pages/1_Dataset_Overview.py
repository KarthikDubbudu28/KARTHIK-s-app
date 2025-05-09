import streamlit as st
import pandas as pd
import requests
from io import StringIO

st.header("📄 Dataset Overview")

@st.cache_data  # Use @st.cache if using an older version of Streamlit
def load_data():
    url = 'https://raw.githubusercontent.com/your-username/your-repo-name/main/beijing_cleaned.csv'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data from GitHub.")
        return None

# Load data
data = load_data()

# Proceed only if data is successfully loaded
if data is not None:
    st.write("Shape:", data.shape)

    if 'station' in data.columns:
        unique_stations = data['station'].unique()
        selected_station = st.selectbox("Select a station to view its data:", unique_stations)
        station_data = data[data['station'] == selected_station]
        st.dataframe(station_data.head(20))
        st.write("Total records for this station:", station_data.shape[0])
    else:
        st.warning("No 'station' column found in dataset.")



