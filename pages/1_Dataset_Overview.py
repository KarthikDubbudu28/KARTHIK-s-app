import streamlit as st
import pandas as pd
import requests
from io import StringIO

st.header("📄 Dataset Overview")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data from GitHub.")
        return None

# Load data
data = load_data()

if data is not None:
    st.write("Shape:", data.shape)

    # Section: Pollutant Categories
    st.subheader("🧪 Pollutant Categories")

    solid_pollutants = ['PM2.5', 'PM10']
    gas_pollutants = ['SO2', 'NO2', 'CO', 'O3']

    st.markdown("**Solid Pollutants:**")
    st.write(", ".join(solid_pollutants))

    st.markdown("**Gas Pollutants:**")
    st.write(", ".join(gas_pollutants))

    # Optional: Show these columns from dataset if they exist
    st.subheader("📊 Sample Pollutant Data (first 5 rows)")
    pollutant_columns = solid_pollutants + gas_pollutants
    available_columns = [col for col in pollutant_columns if col in data.columns]
    
    if available_columns:
        st.dataframe(data[available_columns].head())
    else:
        st.info("No pollutant columns found in dataset.")

    # Section: Station-wise filtering
    if 'station' in data.columns:
        unique_stations = data['station'].unique()
        selected_station = st.selectbox("Select a station to view its data:", unique_stations)
        station_data = data[data['station'] == selected_station]
        st.dataframe(station_data.head(20))
        st.write("Total records for this station:", station_data.shape[0])
    else:
        st.warning("No 'station' column found in dataset.")





