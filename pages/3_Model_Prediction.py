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

# Load dataset
data_url = "https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv"
df = pd.read_csv(data_url)

# Clean data
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
target = 'TEMP'
df.dropna(subset=pollutants + [target], inplace=True)

st.title("🌡️ Temperature Prediction")
st.subheader("Select features, algorithm, and input pollutant levels to predict temperature")

# Step 1: Feature selection
selected_features = st.multiselect("Select pollutants to use as features:", pollutants, default=pollutants)

if len(selected_features) < 1:
    st.warning("Please select at least one pollutant.")
    st.stop()

# Step 2: Input feature values
input_values = {}
st.markdown("### Enter pollutant levels:")
for feature in selected_features:
    input_values[feature] = st.number_input(f"Enter value for {feature}:", value=0)

# Step 3: Algorithm and Grid Search selection
algorithms = ['Support Vector Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost']
selected_algorithms = st.multiselect("Select models to use:", algorithms, default=algorithms)
use_grid_search = st.checkbox("Use Grid Search", value=False)

if st.button("Predict Temperature"):
    X = df[selected_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    user_input_scaled = scaler.transform([list(input_values.values())])

    results = []

    for model_name in selected_algorithms:
        if model_name == 'Support Vector Regression':
            if use_grid_search:
                param_grid = {'C': [1, 10], 'gamma': [0.01], 'epsilon': [0.1]}
                model = GridSearchCV(SVR(kernel='rbf'), param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
            else:
                model = SVR(kernel='rbf')
        elif model_name == 'KNN':
            if use_grid_search:
                param_grid = {'n_neighbors': [3, 5, 7]}
                model = GridSearchCV(KNeighborsRegressor(), param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
            else:
                model = KNeighborsRegressor(n_neighbors=5)
        elif model_name == 'Decision Tree':
            if use_grid_search:
                param_grid = {'max_depth': [None, 5, 10]}
                model = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
            else:
                model = DecisionTreeRegressor(random_state=42)
        elif model_name == 'Random Forest':
            if use_grid_search:
                param_grid = {'n_estimators': [10, 50], 'max_depth': [None, 5]}
                model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_name == 'XGBoost':
            if use_grid_search:
                param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
                model = GridSearchCV(XGBRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
            else:
                model = XGBRegressor(n_estimators=100, random_state=42)

        model.fit(X_train_scaled, y_train)
        prediction = model.predict(user_input_scaled)[0]
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Model': model_name,
            'Predicted TEMP': round(prediction, 2),
            'RMSE': round(rmse, 2),
            'R² Score': round(r2, 2)
        })

    st.success("🎯 Here is your predicted temperature!")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    st.bar_chart(results_df.set_index('Model')[['Predicted TEMP']])
