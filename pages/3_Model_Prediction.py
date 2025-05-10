import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Load the dataset from URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv"  
    df = pd.read_csv(url)
    return df.dropna(subset=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP'])

df = load_data()
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
target = 'TEMP'

st.title("🌡️ Model Prediction Page")
st.subheader("Enter pollutant values to predict temperature using selected ML algorithms")

# --- Sidebar Input ---
selected_pollutants = st.multiselect("Select Pollutants", pollutants, default=pollutants)

if len(selected_pollutants) != len(pollutants):
    st.warning("You must select all pollutants for accurate prediction.")
    st.stop()

user_input = {}
for pollutant in selected_pollutants:
    user_input[pollutant] = st.number_input(f"Enter value for {pollutant}", value=0.0, format="%.2f")

selected_models = st.multiselect(
    "Select Models",
    ['Support Vector Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'],
    default=['Support Vector Regression']
)

use_grid_search = st.checkbox("Use Grid Search to Tune Hyperparameters", value=False)

if st.button("Predict Temperature"):
    X = df[selected_pollutants]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_user_scaled = scaler.transform([list(user_input.values())])

    results = []

    for model_name in selected_models:
        if model_name == "Support Vector Regression":
            model = SVR()
            if use_grid_search:
                param_grid = {'C': [1, 10], 'gamma': [0.01, 0.1], 'epsilon': [0.1, 0.2]}
                model = GridSearchCV(SVR(), param_grid, cv=3, n_jobs=-1)
        elif model_name == "KNN":
            model = KNeighborsRegressor()
            if use_grid_search:
                param_grid = {'n_neighbors': [3, 5, 7]}
                model = GridSearchCV(KNeighborsRegressor(), param_grid, cv=3, n_jobs=-1)
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor()
            if use_grid_search:
                param_grid = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
                model = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=3, n_jobs=-1)
        elif model_name == "Random Forest":
            model = RandomForestRegressor()
            if use_grid_search:
                param_grid = {'n_estimators': [10, 50], 'max_depth': [None, 10]}
                model = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, n_jobs=-1)
        elif model_name == "XGBoost":
            model = XGBRegressor(verbosity=0)
            if use_grid_search:
                param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
                model = GridSearchCV(XGBRegressor(verbosity=0), param_grid, cv=3, n_jobs=-1)

        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_user_scaled)[0]

        # Metrics using test set
        y_pred_test = model.predict(scaler.transform(X_test))
        rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        r2 = r2_score(y_test, y_pred_test)

        results.append({
            'Model': model_name,
            'Predicted TEMP': round(prediction, 2),
            'RMSE': round(rmse, 2),
            'R²': round(r2, 2)
        })

    # --- Display results ---
    st.success("🎯 Here is your predicted temperature based on selected models:")
    result_df = pd.DataFrame(results)
    st.dataframe(result_df)

    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Bar(x=result_df['Model'], y=result_df['Predicted TEMP'], name='Predicted TEMP'))
    fig.add_trace(go.Scatter(x=result_df['Model'], y=result_df['RMSE'], mode='lines+markers', name='RMSE'))
    fig.update_layout(title="Comparison of Model Predictions", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

