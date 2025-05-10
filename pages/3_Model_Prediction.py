import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv'  # Replace with actual URL
    df = pd.read_csv(url)
    return df

# Feature selection and target variable
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
target = 'TEMP'

# Load data and display dataset preview
df = load_data()
st.write("### Dataset Preview")
st.write(df.head())

# User Input for feature values
st.sidebar.header('Enter Values for Prediction')

# Create input fields for user to enter values for pollutants
user_inputs = {
    'PM2.5': st.sidebar.number_input('PM2.5', value=0.0, step=0.1),
    'PM10': st.sidebar.number_input('PM10', value=0.0, step=0.1),
    'SO2': st.sidebar.number_input('SO2', value=0.0, step=0.1),
    'NO2': st.sidebar.number_input('NO2', value=0.0, step=0.1),
    'CO': st.sidebar.number_input('CO', value=0.0, step=0.1),
    'O3': st.sidebar.number_input('O3', value=0.0, step=0.1),
}

# Convert the user inputs to a DataFrame
user_data = pd.DataFrame(user_inputs, index=[0])

# Model Selection
model_choice = st.sidebar.selectbox(
    'Select Model',
    ['SVR', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost']
)

# Grid Search Option
grid_search = st.sidebar.checkbox('Enable Grid Search')

# Load dataset and define features
X = df[features]
y = df[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Dictionary for easy access
models = {
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

# Hyperparameters for Grid Search (if enabled)
param_grid = {
    'SVR': {'C': [1, 10], 'gamma': [0.1, 0.01], 'epsilon': [0.1, 0.2]},
    'KNN': {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']},
    'Decision Tree': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]},
    'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01], 'max_depth': [5, 10], 'subsample': [0.8, 1]}
}

# Fit the selected model
if grid_search:
    from sklearn.model_selection import GridSearchCV

    st.write("Performing Grid Search...")

    model = models[model_choice]
    grid_search_cv = GridSearchCV(estimator=model, param_grid=param_grid[model_choice], cv=3, scoring='neg_mean_squared_error')
    grid_search_cv.fit(X_train_scaled, y_train)

    # Get best parameters and best score
    best_model = grid_search_cv.best_estimator_
    st.write(f"Best Parameters: {grid_search_cv.best_params_}")

else:
    best_model = models[model_choice]
    best_model.fit(X_train_scaled, y_train)

# Make predictions on the test set and user input
y_pred_test = best_model.predict(X_test_scaled)
y_pred_user = best_model.predict(scaler.transform(user_data))

# Calculate RMSE and R² score
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# Display results
st.write(f"### Results for {model_choice}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.2f}")
st.write(f"Predicted Temperature for your input: {y_pred_user[0]:.2f}°C")

# Display comparison of models
model_comparisons = pd.DataFrame({
    'Model': ['SVR', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'R²': [
        r2_score(y_test, models['SVR'].fit(X_train_scaled, y_train).predict(X_test_scaled)),
        r2_score(y_test, models['KNN'].fit(X_train_scaled, y_train).predict(X_test_scaled)),
        r2_score(y_test, models['Decision Tree'].fit(X_train_scaled, y_train).predict(X_test_scaled)),
        r2_score(y_test, models['Random Forest'].fit(X_train_scaled, y_train).predict(X_test_scaled)),
        r2_score(y_test, models['XGBoost'].fit(X_train_scaled, y_train).predict(X_test_scaled))
    ]
})

st.write("### Model Comparison")
st.write(model_comparisons)

# Visualization of the results (Predicted vs Actual values)
st.write("### Prediction vs Actual (Test Set)")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_test)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Actual vs Predicted Temperature')
st.pyplot(fig)

# Display model predictions in a bar chart
st.write("### Model Comparison - R² Scores")
fig = px.bar(model_comparisons, x='Model', y='R²', title="Model Comparison based on R² Score")
st.plotly_chart(fig)


