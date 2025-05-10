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
import plotly.express as px

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv"
    df = pd.read_csv(url)
    df.dropna(subset=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP'], inplace=True)
    return df

df = load_data()
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
target = 'TEMP'

st.sidebar.title("Model Prediction Options")
algorithm = st.sidebar.selectbox("Choose Algorithm", 
                                 ['Support Vector Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'])
use_grid_search = st.sidebar.checkbox("Use Grid Search")

st.title("🔍 Predict Temperature using ML Algorithms")
st.subheader("📊 Enter pollutant values:")

user_input = []
for feature in features:
    val = st.number_input(f"Enter {feature}", format="%.2f")
    user_input.append(val)

if st.button("Predict Temperature"):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    user_input_scaled = scaler.transform([user_input])

    model = None
    params = {}

    if algorithm == 'Support Vector Regression':
        model = SVR()
        if use_grid_search:
            params = {'C': [1, 10], 'gamma': [0.1, 0.01], 'epsilon': [0.1, 0.2]}
    elif algorithm == 'KNN':
        model = KNeighborsRegressor()
        if use_grid_search:
            params = {'n_neighbors': [3, 5, 7]}
    elif algorithm == 'Decision Tree':
        model = DecisionTreeRegressor()
        if use_grid_search:
            params = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
    elif algorithm == 'Random Forest':
        model = RandomForestRegressor(n_jobs=-1)
        if use_grid_search:
            params = {'n_estimators': [50, 100], 'max_depth': [None, 5], 'min_samples_split': [2, 5]}
    elif algorithm == 'XGBoost':
        model = XGBRegressor(n_jobs=-1)
        if use_grid_search:
            params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.05]}

    if use_grid_search and params:
        search = GridSearchCV(model, params, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_
    else:
        model.fit(X_train_scaled, y_train)

    prediction = model.predict(user_input_scaled)[0]
    y_pred_test = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)


    st.success(f"🌡️ Here is your predicted temperature: **{prediction:.2f} °C**")
    st.write(f" RMSE: {rmse:.2f}")
    st.write(f" R² Score: {r2:.2f}")

    # Comparison Table and Bar Chart
    results = [{'Model': algorithm, 'Predicted TEMP': prediction}]
    results_df = pd.DataFrame(results)
    st.subheader("📊 Comparison Table of Predicted Temperature")
    st.dataframe(results_df)
    st.subheader("📈 Temperature Predictions by Model")
    fig = px.bar(results_df, x='Model', y='Predicted TEMP', color='Model', text='Predicted TEMP')
    st.plotly_chart(fig, use_container_width=True)

    # Save prediction history
    st.session_state.prediction_history.append({
        "Input": dict(zip(features, user_input)),
        "Results": results
    })

# Show history
if st.session_state.prediction_history:
    st.subheader("🕒 Prediction History in This Session")
    for i, record in enumerate(st.session_state.prediction_history):
        st.markdown(f"**Prediction {i + 1}:** Input = {record['Input']}")
        df_result = pd.DataFrame(record['Results'])
        st.dataframe(df_result)

if st.button("🧹 Clear Prediction History"):
    st.session_state.prediction_history = []
    st.success("Prediction history cleared.")







