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

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load dataset (cached)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/KarthikDubbudu28/KARTHIK-s-app/refs/heads/main/beijing_cleaned.csv"
    df = pd.read_csv(url)
    df.dropna(subset=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP'], inplace=True)
    return df

df = load_data()
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
target = 'TEMP'

# Sidebar for model selection
st.sidebar.title("Model Prediction Options")
selected_model = st.sidebar.selectbox(
    "Select Model to Predict",
    ['Random Forest', 'KNN', 'Decision Tree', 'Support Vector Regression', 'XGBoost']
)
use_grid_search = st.sidebar.checkbox("Use Grid Search (slower)")

# Main UI
st.title("🌡️ Temperature Prediction using Machine Learning")
st.subheader("✍️ Enter Pollutant Values")

user_input = []
for feature in features:
    val = st.number_input(f"Enter {feature}", format="%.2f")
    user_input.append(val)

if st.button("\U0001F52E Predict Temperature"):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    user_input_scaled = scaler.transform([user_input])

    model = None
    params = {}

    if selected_model == 'Support Vector Regression':
        model = SVR()
        params = {'C': [1], 'gamma': ['scale'], 'epsilon': [0.1]}
    elif selected_model == 'KNN':
        model = KNeighborsRegressor()
        params = {'n_neighbors': [5]}
    elif selected_model == 'Decision Tree':
        model = DecisionTreeRegressor()
        params = {'max_depth': [5]}
    elif selected_model == 'Random Forest':
        model = RandomForestRegressor(n_jobs=-1)
        params = {'n_estimators': [50], 'max_depth': [5]}
    elif selected_model == 'XGBoost':
        model = XGBRegressor(n_jobs=-1, verbosity=0)
        params = {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}

    if use_grid_search:
        search = GridSearchCV(model, params, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_
    else:
        model.set_params(**{k: v[0] for k, v in params.items()})
        model.fit(X_train_scaled, y_train)

    prediction = model.predict(user_input_scaled)[0]
    y_pred_test = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)


    # Show prediction results
    st.success(f"🤔 Predicted Temperature: **{prediction:.2f} °C**")
    st.write(f"😮‍💨 RMSE: **{rmse:.2f}**")
    st.write(f"😰 R² Score: **{r2:.2f}**")
    st.write(f"🧐 Grid Search Used: **{'Yes' if use_grid_search else 'No'}**")

    # Show results table and chart
    results = [{
        'Model': selected_model,
        'Predicted TEMP': round(prediction, 2),
        'RMSE': round(rmse, 2),
        'R² Score': round(r2, 2),
        'Grid Search': 'Yes' if use_grid_search else 'No'
    }]
    results_df = pd.DataFrame(results)

    st.subheader(" Comparison Table")
    st.dataframe(results_df)

    st.subheader("Bar Chart: Predicted TEMP")
    fig = px.bar(results_df, x='Model', y='Predicted TEMP', color='Model', text='Predicted TEMP',
                 title="Predicted Temperature by Selected Algorithm")
    st.plotly_chart(fig, use_container_width=True)

    # Store in session state
    st.session_state.prediction_history.append({
        "Input": dict(zip(features, user_input)),
        "Results": results
    })

# Show prediction history
if st.session_state.prediction_history:
    st.subheader("\U0001F552 Prediction History")
    for i, record in enumerate(st.session_state.prediction_history):
        st.markdown(f"**Prediction {i + 1}:** Input = {record['Input']}")
        st.dataframe(pd.DataFrame(record['Results']))

if st.button("\U0001F9F9 Clear Prediction History"):
    st.session_state.prediction_history = []
    st.success("\u2705 Prediction history cleared.")











