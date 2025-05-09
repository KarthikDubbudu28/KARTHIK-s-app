import streamlit as st
import pandas as pd

st.header("📄 Dataset Overview")

@st.cache_data
def load_data():
    return pd.read_csv("beijing_cleaned.csv")

data = load_data()
st.dataframe(data)
st.write("Shape:", data.shape)

if 'station' in data.columns:
    unique_stations = data['station'].unique()
    selected_station = st.selectbox("Select a station to view its data:", unique_stations)
    station_data = data[data['station'] == selected_station]
    st.dataframe(station_data.head(20))
    st.write("Total records for this station:", station_data.shape[0])
else:
    st.warning("No 'station' column found in dataset.")
