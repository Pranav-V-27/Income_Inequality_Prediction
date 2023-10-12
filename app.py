import streamlit as st
import joblib
import pandas as pd

from prediction_template import get_prediction, label_encoder

# Load the trained model
try:
    model = 'Model/logistic_regression_model.pkl'
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Load the label encoders for categorical features

# Create title and intro
st.title('Income Inequality Prediction App')
st.write("""
This app predicts whether a person's income will be above the limit based on census data.
""")

# Get user input
st.subheader('User Input Parameters')

age = st.number_input('Age', min_value=0, max_value=100, value=25)
#education = st.selectbox('Education', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
education_num = st.number_input('Education Num', min_value=0, max_value=16, value=10)
#marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
#occupation = st.text_input('Occupation')
#race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
#sex = st.selectbox('Sex', ['Female', 'Male'])
#capital_gain = st.number_input('Capital Gain', min_value=0)
#capital_loss = st.number_input('Capital Loss', min_value=0)
#hours_per_week = st.number_input('Hours Per Week', min_value=0)
#native_country = st.text_input('Native Country')

# Create a dictionary with user input data
user_data = {
   'age': age,
   #'education': label_encoder.transform([education])['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],  # Corrected this line
   'education_num': education_num,
   #'marital_status': label_encoder.transform([marital_status])[0],
   #'occupation': label_encoder.transform([occupation])[0],
   #'race': label_encoder.transform([race])[0],
   #'sex': label_encoder.transform([sex])[0],
   #'capital_gain': capital_gain,
   #'capital_loss': capital_loss,
   #'hours_per_week': hours_per_week,
   #'native_country': native_country
}


# Transform into DataFrame
features = pd.DataFrame(user_data, index=[0])

# Make prediction
prediction = model.predict(features)

# Output prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('Income will likely be above the limit')
else:
    st.write('Income will likely not be above the limit')
