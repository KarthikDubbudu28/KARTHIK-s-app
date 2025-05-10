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
import matplotlib.pyplot as plt
import plotly.express as px

# Load the dataset from URL
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv'  # 🔁 Update this URL
    df = pd.read_csv(url)
    return df

# App Title
st.title("🌡️ Temperature Prediction Based on Pollutants")

# Load and preprocess data
df = load_data()
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
target = 'TEMP'
df = df.dropna(subset=features + [target])

# Sidebar inputs
st.sidebar.header("🔢 Input Pollutant Values")
user_inputs = {
    feat: st.sidebar.number_input(f"{feat}", value=0.0, step=0.1, format="%.2f")
    for feat in features
}
user_data = pd.DataFrame([user_inputs])

# Sidebar model selection
st.sidebar.header("🧠 Model Selection")
model_choice = st.sidebar.selectbox("Select a Model", ['SVR', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'])
grid_search_enabled = st.sidebar.checkbox("Enable Grid Search")

# Prepare training data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
user_scaled = scaler.transform(user_data)

# Model definitions
models = {
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

param_grids = {
    'SVR': {'C': [1, 10], 'gamma': [0.1, 0.01], 'epsilon': [0.1]},
    'KNN': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']},
    'Decision Tree': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]},
    'Random Forest': {'n_estimators': [10, 50], 'max_depth': [None, 5]},
    'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.1], 'max_depth': [3, 5]}
}

# 🔘 Predict Button
if st.button("🎯 Predict Temperature"):

    # Train model (with or without grid search)
    if grid_search_enabled:
        st.info("Performing grid search...")
        grid = GridSearchCV(models[model_choice], param_grids[model_choice], cv=3, scoring='neg_mean_squared_error')
        grid.fit(X_train_scaled, y_train)
        model = grid.best_estimator_
        st.success(f"Best parameters: {grid.best_params_}")
    else:
        model = models[model_choice]
        model.fit(X_train_scaled, y_train)

    # Predictions
    y_test_pred = model.predict(X_test_scaled)
    y_user_pred = model.predict(user_scaled)[0]

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    # Results
    st.subheader("📊 Prediction Results")
    st.write(f"**Predicted Temperature**: 🌡️ `{y_user_pred:.2f}°C`")
    st.write(f"**RMSE on Test Set**: `{rmse:.2f}`")
    st.write(f"**R² Score**: `{r2:.2f}`")

    # Comparison chart (optional)
    st.subheader("📈 Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel("Actual TEMP")
    ax.set_ylabel("Predicted TEMP")
    ax.set_title("Actual vs Predicted Temperature")
    st.pyplot(fig)



