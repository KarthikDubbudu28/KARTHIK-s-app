import streamlit as st
import pandas as pd

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





