import streamlit as st
import pickle
import numpy as np
import os
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Define model file paths
MODEL_PATHS = {
    "Gradient Boosting": "Gradient Boosting_nautical_mile_final.pkl",
    "Decision Tree": "Decision Tree_nautical_mile_final.pkl",
    "Random Forest": "Random Forest_nautical_mile_final.pkl",
    "AdaBoost": "AdaBoost_nautical_mile_final.pkl",
    "K-Nearest Neighbors": "K-Nearest Neighbors_nautical_mile_final.pkl",
    "Linear Regression": "Linear Regression_nautical_mile_final.pkl",
    "Bayesian Ridge": "Bayesian Ridge_nautical_mile_final.pkl",
    "Multi-layer Perceptron": "Multi-layer Perceptron_nautical_mile_final.pkl",
}

# Define models for later use
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Bayesian Ridge": BayesianRidge(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Multi-layer Perceptron": MLPRegressor(),
}

# List of expected input features for prediction
EXPECTED_FEATURES = [
    'airpressure', 'airtemperature', 'averagespeedgps', 'averagespeedlog',
    'cargometrictons', 'currentstrength', 'distancefromlastport',
    'distancetonextport', 'distancetravelledsincelastreport',
    'enginedriftingstoppagetime', 'engineroomairpressure',
    'engineroomairtemperature', 'engineroomrelativeairhumidity',
    'engineslip', 'isfuelchangeover', 'isturbochargercutout',
    'relativeairhumidity', 'remainingdistancetoeosp', 'remainingtimetoeosp',
    'scavengingaircoolingwatertemperatureaftercooler', 'scavengingairpressure',
    'scavengingairtemperatureaftercooler', 'seastate', 'seastatedirection',
    'totalcylinderoilconsumption', 'totalcylinderoilspecificconsumption',
    'watertemperature', 'winddirection', 'winddirectionisvariable',
    'tugsused', 'voyagenumber', 'distanceeosptofwe', 'timesteamed',
    'bendingmomentsinpercent', 'dischargedsludge', 'metacentricheight',
    'shearforcesinpercent', 'distancetoeosp', 'saileddistance',
    'runninghourscountervalue', 'energyproducedcountervalue',
    'energyproducedinreportperiod', 'consumption', 'runninghours',
    'new_timezoneinfo_05:30', 'new_timezoneinfo_07:30',
    'new_timezoneinfo_08:30', 'new_timezoneinfo_09:30',
    'new_timezoneinfo_10:30', 'new_timezoneinfo_11:00',
    'new_timezoneinfo_11:30', 'new_timezoneinfo_12:00',
    'new_timezoneinfo_12:30', 'new_timezoneinfo_13:30',
    'new_timezoneinfo_14:30', 'new_timezoneinfo_15:30',
    'new_timezoneinfo_16:30', 'new_timezoneinfo_17:30',
    'new_timezoneinfo_3:30', 'new_timezoneinfo_4:30',
    'new_timezoneinfo_5:30', 'new_timezoneinfo_6:30',
    'new_timezoneinfo_7:30', 'totalconsumption'
]

# Load the models from file
def load_models():
    loaded_models = {}
    for model_name, model_path in MODEL_PATHS.items():
        try:
            # Ensure the model path is correct
            if os.path.exists(model_path):
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                loaded_models[model_name] = model
            else:
                st.error(f"Error loading model {model_name}: Model file not found at {model_path}")
                return None
        except Exception as e:
            st.error(f"Error loading model {model_name}: {e}")
            return None
    return loaded_models

# Estimate fuel consumption
def estimate_fuel_consumption(model_name, airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance, models):
    if models is None:
        st.error("Models are not loaded.")
        return None, None

    if model_name not in models:
        st.error(f"Model {model_name} not found.")
        return None, None

    # Prepare input features for prediction
    x = np.zeros(len(EXPECTED_FEATURES))
    x[0] = airpressure
    x[42] = consumption
    x[24] = totalcylinderoilconsumption
    x[25] = totalcylinderoilspecificconsumption
    x[38] = saileddistance

    model = models[model_name]
    fuel_per_nautical_mile = model.predict([x])[0]
    total_consumption = fuel_per_nautical_mile * saileddistance

    return fuel_per_nautical_mile, total_consumption

# Load all models
models = load_models()

if models:
    # Apply custom CSS styling
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f8ff; /* Light blue background */
        }
        .sidebar .sidebar-content h2 {
            color: #ff4500; /* Red-Orange color */
        }
        h1 {
            color: #4b0082; /* Indigo color */
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Streamlit app title
    st.title("Fuel Consumption Prediction API Using Streamlit")

    # Description for the app
    st.markdown("""
    <div style="text-align: center; background-color: #ffc0cb; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <h1 style="color: #4b0082;">Choose a model and fill the details to get accurate predictions</h1>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for model selection and input parameters
    st.sidebar.header("Select Model")
    model_name = st.sidebar.selectbox("", list(MODEL_PATHS.keys()))
    st.sidebar.header("Input Parameters")

    # Input fields for prediction
    airpressure = st.sidebar.number_input("Air Pressure:", format="%.2f")
    consumption = st.sidebar.number_input("Consumption:", format="%.2f")
    totalcylinderoilconsumption = st.sidebar.number_input("Total Cylinder Oil Consumption:", format="%.2f")
    totalcylinderoilspecificconsumption = st.sidebar.number_input("Total Cylinder Oil Specific Consumption:", format="%.2f")
    saileddistance = st.sidebar.number_input("Sailed Distance:", format="%.2f")

    # Prediction button
    
    if st.sidebar.button("Predict"):
      fuel_per_nautical_mile, total_consumption = estimate_fuel_consumption(
        model_name, airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance, models)
    
        if fuel_per_nautical_mile is not None:
            st.markdown("""
            <div style="text-align: center; background-color: #ffffff; padding: 10px; border-radius: 10px;">
                <h2 style="color: orange;">Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)
    
            st.markdown(f"""
            <div style="
                border: 2px solid #ff4500;
                border-radius: 15px; 
                background-color: #add8e6; 
                padding: 20px; 
                margin: 20px auto; 
                width: 80%; 
                text-align: center;">
                <h3 style="color: red; margin-bottom: 10px;">Fuel Per Nautical Mile ({model_name}):</h3>
                <p style="font-size: 20px; font-weight: bold; color: #ff4500;">{fuel_per_nautical_mile:.2f}</p>
                <h3 style="color: red; margin-top: 10px;">Total Consumption ({model_name}):</h3>
                <p style="font-size: 20px; font-weight: bold; color: #ff4500;">{total_consumption:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            
            st.error("Unable to make predictions. Please check the inputs or model.")

    


           



