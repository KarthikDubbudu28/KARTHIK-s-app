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


elif page == "Explore EDA":
   

    # Insights Summary at the Beginning
    st.markdown("### 📌 Insights from Summary Statistics")

    st.markdown("""
    1. **Timeline:**  
       - The dataset covers the period from **03/01/2013 to 02/28/2017**.  
       - The average date suggests the midpoint is around **03/01/2015**.

    2. **Solid Pollutants (PM2.5 and PM10):**  
       - **PM2.5:** Mean = 78.7 µg/m³  
       - **PM10:** Mean = 102 µg/m³  
       - Both have maximum values close to **999**, indicating **extremely high pollution levels**.

    3. **Gas Pollutants (SO₂, NO₂, CO, O₃):**
       - **CO (Carbon Monoxide):**  
         - Mean = **1202 µg/m³**  
         - Max = **10,000 µg/m³** → **Severe air pollution**

       - **O₃ (Ozone):**  
         - Mean = **56.5 µg/m³**, Max = **674 µg/m³**

       - **NO₂ (Nitrogen Dioxide):**  
         - Mean = **48.9 µg/m³**, Max = **264 µg/m³**  
         - May contribute to **acid rain**

       - **SO₂ (Sulfur Dioxide):**  
         - Mean = **14.85 µg/m³**, Max = **411 µg/m³**  
         - Can cause **climate change effects**
    """)
elif page == "Explore EDA":
   
    # Pollutants Selection
    st.markdown("### 🧪 Pollutant Mean Levels")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    selected_pollutants = st.multiselect(
        "Select pollutants to view their average levels:",
        options=pollutants,
        default=["PM2.5"],  # Default selection to ensure minimum one is selected
        help="Hold Ctrl (Windows) or Command (Mac) to select multiple pollutants."
    )

    if selected_pollutants:
        poll = data[selected_pollutants].mean()
        pollutants_df = poll.to_frame().reset_index()
        pollutants_df.columns = ['Pollutant', 'Level']
        st.dataframe(pollutants_df)
    else:
        st.warning("⚠️ Please select at least one pollutant to display data.")














    
    
