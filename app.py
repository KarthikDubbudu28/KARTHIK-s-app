import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Temperature Prediction App", layout="wide")
st.title("🌡️ Temperature Prediction App")
st.markdown("Welcome to the Temperature Prediction App for Beijing City!")

# Sidebar Navigation
with st.sidebar:
    st.header("📂 Pages")
    page = st.radio("Navigate to:", ["Dataset Overview", "Explore EDA", "Model Prediction"])

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("beijing_cleaned.csv")

data = load_data()

# Dataset Overview Page
if page == "Dataset Overview":
    st.subheader("📄 Dataset Overview")
    st.dataframe(data)
    st.write("Shape:", data.shape)

    st.markdown("### 🧪 Pollutant Breakdown")
    gas_pollutants = ['NO2', 'CO', 'O3', 'SO2']
    solid_pollutants = ['PM2.5', 'PM10']

    gas_columns = [col for col in gas_pollutants if col in data.columns]
    solid_columns = [col for col in solid_pollutants if col in data.columns]

    if gas_columns:
        st.markdown("#### 💨 Gas Pollutants")
        st.dataframe(data[gas_columns].head())
    else:
        st.warning("No gas pollutant columns found.")

    if solid_columns:
        st.markdown("#### 🧱 Solid Pollutants")
        st.dataframe(data[solid_columns].head())
    else:
        st.warning("No solid pollutant columns found.")

    st.markdown("### 🏭 Station-wise Data Overview")
    if 'station' in data.columns:
        unique_stations = data['station'].unique()
        st.write(f"Found {len(unique_stations)} stations.")
        selected_station = st.selectbox("Select a station to view its data:", unique_stations)
        station_data = data[data['station'] == selected_station]
        st.dataframe(station_data.head(20))
        st.write("Total records for this station:", station_data.shape[0])
    else:
        st.warning("No 'station' column found in dataset.")

# Explore EDA Page
elif page == "Explore EDA":
    st.subheader("📊 Exploratory Data Analysis")

    # Summary Insights
    st.markdown("### 📌 Insights from Summary Statistics")
    st.markdown("""
    1. This dataset shows that timeline is from **03/01/2013 to 02/28/2017** based on the minimum and maximum dates, and the mean implies that it is around **03/01/2015**.

    2. **Solid Pollutants**: PM2.5 (Mean: **78.7 µg/m³**), PM10 (Mean: **102 µg/m³**) — both max out at **999 µg/m³**.

    3. **Gas Pollutants**:
    - **CO**: Highest mean over **1202 µg/m³**, max **10000 µg/m³**
    - **O3**: Mean **56.5 µg/m³**, max **674 µg/m³**
    - **NO2**: Mean **48.9 µg/m³**, max **264 µg/m³**
    - **SO2**: Mean **14.85 µg/m³**, max **411 µg/m³**
    """)

    # Average Pollutants Table
    st.markdown("### 🧪 Average Levels of Selected Pollutants")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    selected_pollutants = st.multiselect(
        "Select pollutants to display average levels:",
        options=pollutants,
        default=["PM2.5"]
    )
    if selected_pollutants:
        mean_pollution = data[selected_pollutants].mean().reset_index()
        mean_pollution.columns = ['Pollutant', 'Average Level (µg/m³)']
        st.dataframe(mean_pollution)
    else:
        st.warning("⚠️ Please select at least one pollutant to proceed.")

    # Pie Chart
    st.markdown("### 🥧 Pie Chart of Average Pollutant Levels")
    avg_pollution = data[pollutants].mean()
    colors = ['red', 'orange', 'yellow', 'lightblue', 'lightgreen', 'green']
    explode = [0.1] * len(avg_pollution)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(
        avg_pollution,
        labels=avg_pollution.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        shadow=True
    )
    ax1.set_title("Dominant Pollutants in China (Average Levels)")
    ax1.axis('equal')
    st.pyplot(fig1)

    # Station-wise Dominant Pollutant Table
    st.markdown("### 🏭 Dominant Pollutants by Station")
    areawise_pollution_means = data.groupby('station')[pollutants].mean()
    dominant_pollutant_by_area = areawise_pollution_means.idxmax(axis=1)
    dominant_pollutant_df = dominant_pollutant_by_area.reset_index()
    dominant_pollutant_df.columns = ['Station', 'Dominant Pollutant']
    stations = dominant_pollutant_df['Station'].unique()
    selected_stations = st.multiselect("Select Station(s) to view dominant pollutant:", stations, default=stations[:1])
    if selected_stations:
        filtered_df = dominant_pollutant_df[dominant_pollutant_df['Station'].isin(selected_stations)]
        st.dataframe(filtered_df)
    else:
        st.warning("Please select at least one station to view the data.")

    # Correlation Heatmap
    st.markdown("### 🔗 Correlation Heatmap of Pollutants")
    numeric_pollutants_df = data[pollutants].apply(pd.to_numeric, errors='coerce')
    correlation_matrix = numeric_pollutants_df.corr()

    col1, col2 = st.columns([3, 2])
    with col1:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0, ax=ax2)
        ax2.set_title("Correlation Heatmap of Pollutants")
        st.pyplot(fig2)

    with col2:
        st.markdown("#### 📘 Interpretation Guide")
        st.markdown("""
        **Positive Correlations:**  
        - `0.00 to 0.25`: Less positive correlation  
        - `0.25 to 0.50`: Moderate positive correlation  
        - `0.50 to 0.75`: Strong positive correlation  
        - `0.75 to 1.00`: Very strong positive correlation  

        **Negative Correlations:**  
        - `0.00 to -0.25`: Less negative correlation  
        - `-0.25 to -0.50`: Moderate negative correlation  
        - `-0.50 to -0.75`: Strong negative correlation  
        - `-0.75 to -1.00`: Very strong negative correlation  
        """)

# Model Prediction Page
elif page == "Model Prediction":
    st.subheader("🤖 Model Prediction")
    st.info("This section will include machine learning models (coming soon).")






   









   















    
    
