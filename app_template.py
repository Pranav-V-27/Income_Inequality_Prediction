import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model_filename = 'logistic_regression_model.pkl' 
model = joblib.load(model_filename)

# Function to predict income above limit
def predict_income(data):
    prediction = model.predict(data)
    return prediction

# Streamlit app
def main():
    st.title('Income Inequality Prediction App')

    st.write("This app predicts whether an individual's income is above or below a certain threshold.")

    # Create input widgets for the features
    pickup_hour = st.number_input('Pickup Hour', min_value=0, max_value=23)
    day_of_week = st.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour_of_accident = st.number_input('Hour of Accident', min_value=0, max_value=23)
    accident_cause = st.text_input('Accident Cause')
    num_vehicles_involved = st.number_input('Number of Vehicles Involved', min_value=1)
    vehicle_type = st.text_input('Vehicle Type')
    driver_age = st.number_input('Driver Age', min_value=0)
    accident_area = st.text_input('Accident Area')
    driving_experience = st.number_input('Driving Experience', min_value=0)
    lanes = st.number_input('Lanes', min_value=1)

    # Map day of the week to numeric values
    day_mapping = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }
    day_of_week = day_mapping.get(day_of_week, 0)

    # Create a feature vector from user inputs
    input_data = np.array([pickup_hour, day_of_week, hour_of_accident, accident_cause, num_vehicles_involved, vehicle_type,
                        driver_age, accident_area, driving_experience, lanes]).reshape(1, -1)

    # Predict the income status
    if st.button('Predict Income Status'):
        prediction = predict_income(input_data)
        if prediction[0] == 1:
            st.write("Predicted Income Status: Above Limit")
        else:
            st.write("Predicted Income Status: Below Limit")

    st.sidebar.markdown("### Project Information")
    st.sidebar.write("This app is a part of the Income Inequality Prediction project.")
    st.sidebar.write("For more details, visit the project's repository on GitHub.")

if __name__ == "__main__":
    main()
