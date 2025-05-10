
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading pre-trained models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load pre-trained models using caching ---
@st.cache_resource
def load_models():
    svr = joblib.load('svr_model.pkl')
    knn = joblib.load('knn_model.pkl')
    dt = joblib.load('dt_model.pkl')
    rf = joblib.load('rf_model.pkl')
    xgb = joblib.load('xgb_model.pkl')
    return svr, knn, dt, rf, xgb

# --- Step 2: Load and cache data ---
@st.cache_data
def load_data():
    # If using an online URL to load data, replace this with your URL
    url = "https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# --- Step 3: Prepare Features and Target ---
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
target = 'TEMP'

# Select columns and clean data
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# --- Step 4: User Input --- 
st.title("Temperature Prediction")
st.write("Enter pollutant values to predict temperature.")

# User input for each pollutant (float values)
pollutant_values = {}
for feature in features:
    pollutant_values[feature] = st.number_input(f"Enter {feature} value", value=0.0)

# --- Step 5: Create DataFrame for prediction ---
input_data = pd.DataFrame([pollutant_values])

# --- Step 6: Preprocessing --- 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale existing data
input_scaled = scaler.transform(input_data)  # Scale user input

# --- Step 7: Select Model ---
selected_model = st.selectbox(
    "Select a model",
    ("Support Vector Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "XGBoost")
)

# --- Step 8: Load pre-trained models ---
svr, knn, dt, rf, xgb = load_models()

# --- Step 9: Predict based on selected model ---
if selected_model == "Support Vector Regression":
    model = svr
elif selected_model == "K-Nearest Neighbors":
    model = knn
elif selected_model == "Decision Tree":
    model = dt
elif selected_model == "Random Forest":
    model = rf
else:
    model = xgb

# --- Step 10: Predict Temperature ---
predicted_temp = model.predict(input_scaled)

# Display prediction result
st.write(f"Predicted Temperature: {predicted_temp[0]:.2f}°C")

# --- Step 11: Display Efficiency Comparison (Optional) ---
# For comparison, display results for all models
model_names = ["SVR", "KNN", "DT", "RF", "XGB"]
predictions = []

for model_name, model in zip(model_names, [svr, knn, dt, rf, xgb]):
    pred = model.predict(input_scaled)
    predictions.append(pred[0])

# Create a DataFrame to show predictions from all models
comparison_df = pd.DataFrame({
    "Model": model_names,
    "Predicted Temperature (°C)": predictions
})

st.write("### Temperature Prediction Comparison")
st.dataframe(comparison_df)

# --- Step 12: Visualize Comparison ---
fig, ax = plt.subplots()
sns.barplot(x=comparison_df["Model"], y=comparison_df["Predicted Temperature (°C)"], ax=ax)
ax.set_title("Predicted Temperature by Different Models")
st.pyplot(fig)

