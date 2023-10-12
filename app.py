import streamlit as st
import joblib
import pandas as pd
from prediction_template import get_prediction, label_encoder

# Load the trained model
model = joblib.load('Model/logistic_regression_model.pkl')

# Load the label encoders for categorical features
label_encoders = {
    'education': joblib.load('Model/education_label_encoder.pkl'),
    'marital_status': joblib.load('Model/marital_status_label_encoder.pkl'),
    'occupation': joblib.load('Model/occupation_label_encoder.pkl'),
    'race': joblib.load('Model/race_label_encoder.pkl'),
    'sex': joblib.load('Model/sex_label_encoder.pkl')
}

# Create title and intro
st.title('Income Inequality Prediction App')
st.write("""
This app predicts whether a person's income will be above the limit based on census data.
""")

# Get user input
st.subheader('User Input Parameters')

age = st.number_input('Age', min_value=0, max_value=100, value=25)
education = st.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
education_num = st.number_input('Education Num', min_value=0, max_value=16, value=10)
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.text_input('Occupation')
race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
sex = st.selectbox('Sex', ['Female', 'Male'])
capital_gain = st.number_input('Capital Gain', min_value=0)
capital_loss = st.number_input('Capital Loss', min_value=0)
hours_per_week = st.number_input('Hours Per Week', min_value=0)
#native_country = st.text_input('Native Country')

# Create a dictionary with user input data
user_data = {
    'age': age,
    'education': education,
    'education_num': education_num,
    'marital_status': marital_status,
    'occupation': occupation,
    'race': race,
    'sex': sex,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'hours_per_week': hours_per_week,
    #'native_country': native_country
}

# Convert user input data into a DataFrame
user_data_df = pd.DataFrame(user_data, index=[0])

# Make prediction using the get_prediction function
prediction = get_prediction(user_data_df, model, label_encoders)

# Output prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('Income will likely be above the limit')
else:
    st.write('Income will likely not be above the limit')
