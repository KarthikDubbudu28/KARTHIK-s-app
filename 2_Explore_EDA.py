import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.header("📊 Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("beijing_cleaned.csv")

data = load_data()

st.markdown("### 📌 Summary Insights")
st.write(data.describe())

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
selected_pollutants = st.multiselect("Select pollutants to show average levels:", pollutants, default=["PM2.5"])

if selected_pollutants:
    avg_pollution = data[selected_pollutants].mean().reset_index()
    avg_pollution.columns = ['Pollutant', 'Average Level']
    st.dataframe(avg_pollution)

# Pie chart
st.markdown("### 🥧 Average Pollutant Levels")
avg_pollution = data[pollutants].mean()
colors = ['red', 'orange', 'yellow', 'lightblue', 'lightgreen', 'green']
explode = [0.1]*len(avg_pollution)
fig, ax = plt.subplots()
ax.pie(avg_pollution, labels=avg_pollution.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
st.pyplot(fig)

# Correlation heatmap
st.markdown("### 🔗 Correlation Heatmap")
numeric_df = data[pollutants].apply(pd.to_numeric, errors='coerce')
corr_matrix = numeric_df.corr()
col1, col2 = st.columns([3, 2])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
with col2:
    st.markdown("#### Interpretation Guide")
    st.markdown("""
**Positive Correlations:**  
- 0.00 to 0.25: Less positive  
- 0.25 to 0.50: Moderate positive  
- 0.50 to 0.75: Strong positive  
- 0.75 to 1.00: Very strong positive  

**Negative Correlations:**  
- 0.00 to -0.25: Less negative  
- -0.25 to -0.50: Moderate negative  
- -0.50 to -0.75: Strong negative  
- -0.75 to -1.00: Very strong negative  
""")

# Dominant pollutant per station
st.markdown("### 🏭 Dominant Pollutant by Station")
station_means = data.groupby("station")[pollutants].mean()
dominant = station_means.idxmax(axis=1).reset_index()
dominant.columns = ['Station', 'Dominant Pollutant']
stations = dominant['Station'].unique()
selected = st.multiselect("Select Station(s):", stations, default=stations[:1])
if selected:
    st.dataframe(dominant[dominant['Station'].isin(selected)])
else:
    st.warning("Please select at least one station.")
