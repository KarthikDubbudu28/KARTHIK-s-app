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

if page == "Dataset Overview":


    st.markdown("### 🧪 Pollutant Breakdown")

    # Manually define pollutant columns (update based on your dataset)
    gas_pollutants = ['NO2', 'CO', 'O3', 'SO2']
    solid_pollutants = ['PM2.5', 'PM10']

    # Check which columns are actually in the dataset
    gas_columns = [col for col in gas_pollutants if col in data.columns]
    solid_columns = [col for col in solid_pollutants if col in data.columns]

    # Display gas pollutants
    if gas_columns:
        st.markdown("#### 💨 Gas Pollutants")
        st.dataframe(data[gas_columns].head())
    else:
        st.warning("No gas pollutant columns found.")

    # Display solid pollutants
    if solid_columns:
        st.markdown("#### 🧱 Solid Pollutants")
        st.dataframe(data[solid_columns].head())
    else:
        st.warning("No solid pollutant columns found.")

    st.markdown("### 🏭 Station-wise Data Overview")

    # Check if 'station' column exists (replace with actual name if different)
    if 'station' in data.columns:
        # Show unique station names
        unique_stations = data['station'].unique()
        st.write(f"Found {len(unique_stations)} stations.")

        # Let user select a station to view
        selected_station = st.selectbox("Select a station to view its data:", unique_stations)

        # Filter data for the selected station
        station_data = data[data['station'] == selected_station]

        # Display filtered table
        st.dataframe(station_data.head(20))  # showing only first 20 rows for clarity
        st.write("Total records for this station:", station_data.shape[0])
    else:
        st.warning("No 'station' column found in dataset.")








    
    
