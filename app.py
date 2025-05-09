import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    

    # Display Summary Insights
    st.markdown("### 📌 Insights from Summary Statistics")
    st.markdown("""
    1. This dataset shows that timeline is from **03/01/2013 to 02/28/2017** based on the minimum and maximum dates, and the mean implies that it is around **03/01/2015**.

    2. **Solid Pollutants** which include **PM2.5** and **PM10**:
    - PM2.5 has a mean of **78.7 µg/m³**.
    - PM10 has a mean of **102 µg/m³**.
    - Both have a max value of **999 µg/m³**, indicating **extremely high pollution**.

    3. **Gas Pollutants**: **SO2, NO2, CO, O3**
    - **CO (Carbon Monoxide)**: Highest mean over **1202 µg/m³**, max of **10000 µg/m³**.
    - **O3 (Ozone)**: Mean of **56.5 µg/m³**, max **674 µg/m³**.
    - **NO2 (Nitrogen Dioxide)**: Mean of **48.9 µg/m³**, max **264 µg/m³**.
    - **SO2 (Sulfur Dioxide)**: Mean of **14.85 µg/m³**, max **411 µg/m³**.
    """)

    # Pollutants Selection
    st.markdown("### 🧪 Average Levels of Selected Pollutants")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    selected_pollutants = st.multiselect(
        label="Select pollutants to display average levels:",
        options=pollutants,
        default=["PM2.5"],  # Ensures at least one is shown by default
        help="Hold Ctrl (Windows) or Command (Mac) to select multiple"
    )

    if selected_pollutants:
        mean_pollution = data[selected_pollutants].mean().reset_index()
        mean_pollution.columns = ['Pollutant', 'Average Level (µg/m³)']
        st.dataframe(mean_pollution)
    else:
        st.warning("⚠️ Please select at least one pollutant to proceed.")

import matplotlib.pyplot as plt

# Pie Chart of Average Pollutants
st.markdown("### 🥧 Pie Chart of Average Pollutant Levels")

# Step 1: Calculate average of each pollutant
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
avg_pollution = data[pollutants].mean()

# Step 2: Colors — red for dominant, green for least
colors = ['red', 'orange', 'yellow', 'lightblue', 'lightgreen', 'green']

# Step 3: Explode each slice slightly for clarity
explode = [0.1] * len(avg_pollution)

# Step 4: Create Pie Chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(
    avg_pollution,
    labels=avg_pollution.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=explode,
    shadow=True
)
ax.set_title("Dominant Pollutants in China (Average Levels)")
ax.axis('equal')  # Makes the pie chart circular

# Step 5: Display in Streamlit
st.pyplot(fig)

# Station-wise dominant pollutant table
st.markdown("### 🏭 Dominant Pollutants by Station")

# Group and compute dominant pollutants
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
areawise_pollution_means = data.groupby('station')[pollutants].mean()
dominant_pollutant_by_area = areawise_pollution_means.idxmax(axis=1)
dominant_pollutant_df = dominant_pollutant_by_area.reset_index()
dominant_pollutant_df.columns = ['Station', 'Dominant Pollutant']

# Select station(s)
stations = dominant_pollutant_df['Station'].unique()
selected_stations = st.multiselect("Select Station(s) to view dominant pollutant:", stations, default=stations[:1])

# Ensure at least one is selected
if selected_stations:
    filtered_df = dominant_pollutant_df[dominant_pollutant_df['Station'].isin(selected_stations)]
    st.dataframe(filtered_df)
else:
    st.warning("Please select at least one station to view the data.")

import seaborn as sns
import matplotlib.pyplot as plt

# Only include this if it's not already part of your script
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
numeric_pollutants_df = data[pollutants].apply(pd.to_numeric, errors='coerce')
correlation_matrix = numeric_pollutants_df.corr()

st.markdown("### 🔗 Correlation Heatmap of Pollutants")
col1, col2 = st.columns([3, 2])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Pollutants")
    st.pyplot(fig)

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





   









   















    
    
