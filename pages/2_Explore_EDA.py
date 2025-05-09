import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.header("📊 Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("beijing_cleaned.csv")

data = load_data()

st.markdown("### 📌 Summary Insights")
st.write(data.describe())

st.markdown("### 📝 Key Observations and Insights")

st.markdown("""
1. **Dataset Timeline:**  
   This dataset spans from **03/01/2013 to 02/28/2017**. Based on the minimum, maximum, and mean dates, the **average observation date is around 03/01/2015**.

2. **Solid Pollutants** (Particulate Matter):
   - **PM2.5**: Mean = **78.7 µg/m³**
   - **PM10**: Mean = **102 µg/m³**
   - Both have a **maximum concentration of 999 µg/m³**, which is close to the upper measurement limit, indicating **extremely high pollution levels** caused by solid particles.

3. **Gas Pollutants**:
   - **CO (Carbon Monoxide)**:  
     - Mean = **1202 µg/m³**, Max = **10000.0 µg/m³**  
     - Indicates **severe air pollution** potential.
   - **O3 (Ozone)**:  
     - Mean = **56.5 µg/m³**, Max = **674 µg/m³**
   - **NO2 (Nitrogen Dioxide)**:  
     - Mean = **48.9 µg/m³**, Max = **264 µg/m³**  
     - Potential to contribute to **acid rain**.
   - **SO2 (Sulfur Dioxide)**:  
     - Mean = **14.85 µg/m³**, Max = **411 µg/m³**  
     - Can significantly impact **climate change**.
""")

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

# ---- Year-wise Total Pollution Bar Chart ----
st.markdown("### 📈 Year-wise Total Pollution Analysis")

if 'year' in data.columns:
    data['year'] = data['year'].astype(int)

    pollutant_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    available_pollutants = [col for col in pollutant_cols if col in data.columns]

    if available_pollutants:
        yearly_avg = data.groupby('year')[available_pollutants].mean().reset_index()
        yearly_avg['Total_Pollution'] = yearly_avg[available_pollutants].sum(axis=1)
        yearly_avg['Dominant_Pollutant'] = yearly_avg[available_pollutants].idxmax(axis=1)

        yearly_avg = yearly_avg.sort_values(by='Total_Pollution', ascending=True).reset_index(drop=True)

        color_scale = ['green', 'lightgreen', 'yellow', 'orange', 'darkred', 'red']
        num_years = len(yearly_avg)
        yearly_avg['Color'] = pd.cut(
            yearly_avg['Total_Pollution'],
            bins=num_years,
            labels=color_scale[:num_years]
        ).astype(str)

        # Year selection
        selected_years = st.multiselect(
            "Select year(s) to display:",
            options=yearly_avg['year'].tolist(),
            default=[yearly_avg['year'].min()]
        )

        if selected_years:
            filtered_data = yearly_avg[yearly_avg['year'].isin(selected_years)]

            fig = px.bar(
                filtered_data,
                x='year',
                y='Total_Pollution',
                color='Color',
                color_discrete_map={c: c for c in color_scale},
                title='Year-wise Total Pollution with Dominant Pollutant',
                labels={'Total_Pollution': 'Total Pollution (µg/m³)', 'year': 'Year'},
                hover_data=['Dominant_Pollutant']
            )

            fig.update_traces(
                text=filtered_data['Dominant_Pollutant'],
                textposition='outside',
                marker_line_color='black',
                marker_line_width=1
            )
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one year.")
    else:
        st.error("No pollutant columns found for plotting.")
else:
    st.error("Column 'year' is missing from the dataset.")

st.markdown("### 📍 Top 5 Stations by Pollutant Levels")

# Define pollutants
pollutant_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
available_pollutants = [col for col in pollutant_columns if col in data.columns]

# User selection
selected_pollutants = st.multiselect(
    "Select pollutant(s) to view top affected stations:",
    options=available_pollutants,
    default=[available_pollutants[0]]
)

if selected_pollutants:
    mean_pollutant_by_station = data.groupby('station')[available_pollutants].mean()

    for pollutant in selected_pollutants:
        top5 = mean_pollutant_by_station[pollutant].sort_values(ascending=False).head(5)

        st.markdown(f"#### 🔬 Top 5 Stations by **{pollutant}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top5.index, top5.values, color='skyblue')
        ax.set_xlabel(f"{pollutant} Concentration (µg/m³)")
        ax.set_title(f"Top 5 Stations with Highest {pollutant}")
        ax.invert_yaxis()
        st.pyplot(fig)
else:
    st.warning("⚠️ Please select at least one pollutant to display top stations.")


