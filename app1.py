import streamlit as st
import pickle
import numpy as np
import requests
import io

MODEL_PATHS = {
    "Gradient Boosting": "https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/Gradient%20Boosting_nautical_mile_final.pkl",
    "Decision Tree": "https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/Decision%20Tree_nautical_mile_final.pkl",
    "Random Forest": "https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/Random%20Forest_nautical_mile_final.pkl",
    "AdaBoost": "https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/AdaBoost_nautical_mile_final.pkl",
    "K-Nearest Neighbors":"https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/K-Nearest%20Neighbors_nautical_mile_final.pkl",
    "Linear Regression":"https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/Linear%20Regression_nautical_mile_final.pkl",
    "Bayesian Ridge":"https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/Bayesian%20Ridge_nautical_mile_final.pkl",
    "Multi-layer Perceptron":"https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/Multi-layer%20Perceptron_nautical_mile_final.pkl",
    "Stochastic Gradient Descent":"https://github.com/MANKALANIKHILKUMAR/Vessel-Performance-Model-Using-Streamlit-App/raw/refs/heads/main/Stochastic%20Gradient%20Descent_nautical_mile_final.pkl"
    
}


def load_models():
    models = {}
    for model_name, model_url in MODEL_PATHS.items():
        try:
           
            response = requests.get(model_url)
            if response.status_code == 200:
                model_data = io.BytesIO(response.content)
                models[model_name] = pickle.load(model_data)
            else:
                st.error(f"Error loading model {model_name}: Status code {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error loading model {model_name}: {e}")
            return None
    return models

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

def estimate_fuel_consumption(model_name, airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance, models):
    if models is None:
        st.error("Models are not loaded.")
        return None, None

    if model_name not in models:
        st.error(f"Model {model_name} not found.")
        return None, None

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


models = load_models()

if models:
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

    st.title("Fuel Consumption Prediction API Using Streamlit")

    st.markdown("""
    <div style="text-align: center; background-color: #ffc0cb; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <h1 style="color: #4b0082;">Choose a model and fill the details to get accurate predictions</h1>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("Select Model")
    model_name = st.sidebar.selectbox("", list(MODEL_PATHS.keys()))
    st.sidebar.header("Input Parameters")

    airpressure = st.sidebar.number_input("Air Pressure:", format="%.2f")
    consumption = st.sidebar.number_input("Consumption:", format="%.2f")
    totalcylinderoilconsumption = st.sidebar.number_input("Total Cylinder Oil Consumption:", format="%.2f")
    totalcylinderoilspecificconsumption = st.sidebar.number_input("Total Cylinder Oil Specific Consumption:", format="%.2f")
    saileddistance = st.sidebar.number_input("Sailed Distance:", format="%.2f")

    if st.sidebar.button("Predict"):
        fuel_per_nautical_mile, total_consumption = estimate_fuel_consumption(
            model_name, airpressure, consumption, totalcylinderoilconsumption, totalcylinderoilspecificconsumption, saileddistance, models
        )

        if fuel_per_nautical_mile is not None:
            st.markdown("""
            <div style="text-align: center; background-color: #ffffff; padding: 10px; border-radius: 10px;">
                <h2 style="color: red;">Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"**Fuel Per Nautical Mile ({model_name}):** {fuel_per_nautical_mile:.2f}")
            st.write(f"**Total Consumption ({model_name}):** {total_consumption:.2f}")
        else:
            st.error("Unable to make predictions. Please check the inputs or model.")



