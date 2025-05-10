import streamlit as st
import pandas as pd
import streamlit as st

st.subheader("📍 Monitoring Site Classification Justification")

st.markdown("""
In this study, the air quality monitoring stations across Beijing were categorized based on their geographic and socio-environmental characteristics:

- **Wanshouxigong** was classified as **Urban** due to its central location in downtown Beijing, commonly used in studies analyzing urban air pollution patterns ([ACP, 2018](https://acp.copernicus.org/articles/18/6771/2018/)).
- **Wanliu**, located in the northwestern Haidian District, is considered **Suburban**, consistent with its classification in local administrative maps.
- **Dingling**, a Ming tomb site situated in Changping District's northern countryside, was labeled **Rural**, reflecting its low-density, green-space surroundings ([Wikipedia](https://en.wikipedia.org/wiki/Ding_Mausoleum)).
- **Shunyi** was identified as **Industrial** due to its known industrial development zones and manufacturing hubs ([Beijing Government Portal](https://english.beijing.gov.cn/investinginbeijing/WhyBeijing/DistrictsParks/Shunyi/)).
- **Tiantan (Temple of Heaven)** was marked as a **Hotspot** due to its cultural significance and large tourist footfall, being a UNESCO World Heritage Site ([UNESCO](https://whc.unesco.org/en/list/881)).


""")



st.header("📂 Dataset Overview")

# Replace this with your actual dataset URL
DATA_URL = "https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

data = load_data()

# Show column names to confirm the datetime column (useful for debugging)
# st.write("Columns in dataset:", data.columns.tolist())

# Set your datetime column name here (based on your dataset)
datetime_column = 'datetime'  # Change this if needed

if datetime_column in data.columns:
    data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
    data['year'] = data[datetime_column].dt.year

    temp_data = data[['TEMP', 'year']].dropna()

    max_temp_row = temp_data[temp_data['TEMP'] == temp_data['TEMP'].max()].iloc[0]
    min_temp_row = temp_data[temp_data['TEMP'] == temp_data['TEMP'].min()].iloc[0]

    extremes_df = pd.DataFrame({
        "Type": ["🌡️ Highest Temperature", "❄️ Lowest Temperature"],
        "Temperature (°C)": [max_temp_row['TEMP'], min_temp_row['TEMP']],
        "Year": [int(max_temp_row['year']), int(min_temp_row['year'])]
    })

    st.markdown("### 🔥 Temperature Extremes by Year")
    st.table(extremes_df)
else:
    st.error(f"Column '{datetime_column}' not found in dataset. Check the dataset's column names.")

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





